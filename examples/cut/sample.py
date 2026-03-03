#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Sample (inference) script for a trained CUT model on any MAVIC-T task.

Usage (pretrained directory — recommended)::

    python -m examples.cut.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/cut/sar2ir/checkpoint-epoch-400 \
        --split test \
        --output_dir ./samples/cut_sar2ir \
        --batch_size 32

Use ``--batch_size`` to control inference batch size (default 32).

Usage (legacy ``.pt`` file)::

    python -m examples.cut.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./outputs/cut_sar2ir/netG_epoch_400.pt \
        --split test \
        --output_dir ./samples/cut_sar2ir

When a directory is provided the script loads the generator via
``from_pretrained`` following the HuggingFace *diffusers* convention.
Legacy ``.pt`` / ``.safetensors`` single-file checkpoints are still
supported for backward compatibility.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from .dataset_wrapper import MavicTCUTDataset  # noqa: E402
from src.models.cut_model import CUTGenerator, create_generator  # noqa: E402
from src.pipelines.cut import CUTPipeline  # noqa: E402
from src.utils.readme_utils import (  # noqa: E402
    load_checkpoint_config,
    build_detailed_description,
    write_readme,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def _get_output_name(dataset_obj, idx: int) -> str:
    """Resolve submission filename for index, preserving Subset mapping."""
    if isinstance(dataset_obj, Subset):
        base_dataset = dataset_obj.dataset
        base_idx = dataset_obj.indices[idx]
    else:
        base_dataset = dataset_obj
        base_idx = idx

    if hasattr(base_dataset, "get_output_name"):
        return base_dataset.get_output_name(base_idx)

    records = getattr(base_dataset, "_records", None)
    if records is not None:
        stem = Path(records[base_idx]["input_path"]).stem
        return f"{stem}.png"
    return f"sample_{base_idx:05d}.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained CUT generator.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a diffusers-style checkpoint directory or a legacy .pt/.safetensors file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override inference resolution (default: task config resolution).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (CUT is deterministic by default in eval mode).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose output file already exists (default: True, for resumability).",
    )
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--extra_data",
        action="store_true",
        help="Set to 1 in readme when extra data is used.",
    )
    parser.add_argument(
        "--readme_description",
        type=str,
        default="",
        help="Optional custom text appended to the detailed readme description.",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="+",
        default=None,
        help="Device(s) for inference. Single: 'cuda:0'. Multi-GPU: 'cuda:0' 'cuda:1'. Uses DataParallel for multi-GPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()
    # Normalize device: default single device, or list when multi-GPU
    if args.device is None:
        args.device = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    args.primary_device = args.device[0] if isinstance(args.device, list) else args.device
    args.use_multi_gpu = len(args.device) > 1 and all(d.startswith("cuda") for d in args.device)
    return args


def _load_pipeline(
    pretrained_path: str,
    cfg: TaskConfig,
    primary_device: str,
    device_ids: list[int] | None = None,
) -> CUTPipeline:
    """Load the CUT pipeline from a pretrained directory or legacy file."""
    path = Path(pretrained_path)

    if path.is_dir():
        # ---- diffusers from_pretrained path ----
        logger.info("Loading pipeline from pretrained directory: %s", path)
        pipeline = CUTPipeline.from_pretrained(
            pretrained_path, torch_dtype=torch.bfloat16
        )
    else:
        # ---- legacy single-file checkpoint ----
        logger.info("Loading generator from legacy checkpoint: %s", path)
        netG = create_generator(
            input_nc=cfg.source_channels,
            output_nc=cfg.target_channels,
            ngf=cfg.ngf,
            netG=cfg.netG,
            norm_type=cfg.normG,
            use_dropout=not cfg.no_dropout,
            no_antialias=cfg.no_antialias,
            no_antialias_up=cfg.no_antialias_up,
            init_type=cfg.init_type,
            init_gain=cfg.init_gain,
        )
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(str(path))
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        netG.load_state_dict(ckpt)
        pipeline = CUTPipeline(generator=netG)

    pipeline = pipeline.to(primary_device, dtype=torch.bfloat16)
    if device_ids is not None and len(device_ids) > 1:
        pipeline.generator = torch.nn.DataParallel(pipeline.generator, device_ids=device_ids)
        logger.info("Using DataParallel on GPUs %s", device_ids)
    return pipeline


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_ids = None
    if args.use_multi_gpu:
        device_ids = [
            int(d.split(":")[1]) if ":" in d else i
            for i, d in enumerate(args.device)
        ]
    pipeline = _load_pipeline(
        args.pretrained_model_name_or_path,
        cfg,
        args.primary_device,
        device_ids=device_ids,
    )

    # Load evaluation data
    infer_resolution = args.resolution if args.resolution is not None else cfg.resolution
    dataset = MavicTCUTDataset(
        task=args.task,
        split=args.split,
        resolution=infer_resolution,
        source_channels=cfg.source_channels,
        target_channels=cfg.target_channels,
        model_channels=cfg.model_channels,
        with_target=False,
    )
    if args.skip_existing:
        indices_to_process = [
            i for i in range(len(dataset))
            if not (output_dir / _get_output_name(dataset, i)).exists()
        ]
        n_total = len(dataset)
        if not indices_to_process:
            logger.info("All %d outputs already exist in %s. Nothing to do.", n_total, output_dir)
            return
        if len(indices_to_process) < n_total:
            dataset = Subset(dataset, indices_to_process)
            logger.info("Skipping %d existing, processing %d remaining.", n_total - len(indices_to_process), len(indices_to_process))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    det_info = " (deterministic)" if args.deterministic else ""
    logger.info(f"Generating samples for {args.task} ({args.split}), {len(dataset)} inputs{det_info} …")
    all_samples = []
    sample_idx = 0
    total_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling"):
            source = batch[0].to(args.primary_device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(source_image=source, output_type="pt")
            images = result.images

            # Convert from [-1, 1] to [0, 255] uint8
            images_uint8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            images_uint8 = images_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for i, img_arr in enumerate(images_uint8):
                out_name = _get_output_name(dataset, sample_idx + i)
                if img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(2)
                img = Image.fromarray(img_arr)
                img.save(output_dir / out_name)
            sample_idx += len(images_uint8)
            all_samples.append(images_uint8)

    total_elapsed = time.perf_counter() - total_start
    runtime_per_image = total_elapsed / len(dataset) if len(dataset) > 0 else 0.0
    use_gpu = args.primary_device.startswith("cuda")

    checkpoint_config = load_checkpoint_config(args.pretrained_model_name_or_path)
    extra_sampling = [
        "CUT: single-step deterministic generator forward (no diffusion steps)",
    ]
    detailed_description = build_detailed_description(
        model_name="CUT",
        model_description="CUT (Contrastive Unpaired Translation) - contrastive unpaired image-to-image translation. GAN-based, deterministic inference. PyTorch implementation.",
        checkpoint_path=args.pretrained_model_name_or_path,
        args=args,
        cfg=cfg,
        checkpoint_config=checkpoint_config,
        runtime_per_image=runtime_per_image,
        extra_sampling_lines=extra_sampling,
    )
    write_readme(
        output_dir,
        runtime_per_image=runtime_per_image,
        use_gpu=use_gpu,
        extra_data=args.extra_data,
        description=detailed_description,
    )

    all_samples = np.concatenate(all_samples, axis=0)

    if args.save_npz:
        np.savez(output_dir / f"samples_{len(all_samples)}.npz", arr_0=all_samples)
        logger.info(f"Saved NPZ with {len(all_samples)} samples.")

    logger.info(
        f"Sampling complete – {sample_idx} images saved to {output_dir} "
        f"(runtime: {total_elapsed:.1f}s total, {runtime_per_image:.2f}s/image)"
    )


if __name__ == "__main__":
    main()
