#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Sample (inference) script for a trained DBIM model on MAVIC-T tasks.

Outputs are saved under MACIV-T-2025-Submissions/<task>/<model_name>/ following
the official submission format (see official-docs/submission.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.unet_dbim import DBIMUNet  # noqa: E402
from src.pipelines.dbim import DBIMPipeline  # noqa: E402
from src.schedulers import DBIMScheduler  # noqa: E402

from .config import (  # noqa: E402
    TaskConfig,
    rgb2ir_config,
    sar2eo_config,
    sar2ir_config,
    sar2rgb_config,
)
from .dataset_wrapper import MavicTDBIMDataset  # noqa: E402
from src.utils.paths import path_from_root  # noqa: E402
from src.utils.readme_utils import (  # noqa: E402
    load_checkpoint_config,
    build_detailed_description,
    write_readme,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SUBMISSION_ROOT = path_from_root("datasets/BiliSakura/MACIV-T-2025-Submissions")

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


class _SubsetWithOutputNames(Subset):
    """Subset that delegates get_output_name to underlying dataset indices."""

    def get_output_name(self, idx: int) -> str:
        return self.dataset.get_output_name(self.indices[idx])


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained DBIM model.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a diffusers-style checkpoint directory.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory. Default: {DEFAULT_SUBMISSION_ROOT}/<task>/<model_name>/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dbim",
        help="Model name for folder structure. Output: <submission_root>/<task>/<model_name>/",
    )
    parser.add_argument(
        "--submission_root",
        type=str,
        default=str(DEFAULT_SUBMISSION_ROOT),
        help="Root directory for submissions.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance scale (1.0 disables CFG).",
    )
    parser.add_argument("--churn_step_ratio", type=float, default=0.33)
    parser.add_argument("--sampler", type=str, default="dbim", choices=["dbim", "dbim_high_order", "heun"])
    parser.add_argument("--eta", type=float, default=1.0, help="Used by dbim sampler.")
    parser.add_argument("--order", type=int, default=2, choices=[2, 3], help="Used by dbim_high_order.")
    parser.add_argument(
        "--lower_order_final",
        action="store_true",
        default=True,
        help="Use lower order at final step for dbim_high_order (default: True).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (churn_step_ratio=0, eta=0).",
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        default=False,
        help="Clamp denoised predictions to [-1, 1] during sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="+",
        default=None,
        help="Single device (cuda:0) or multi-GPU list (cuda:0 cuda:1).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override input loading resolution. Defaults to task config resolution.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose output file already exists (default: True).",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing output files.",
    )
    parser.add_argument("--save_npz", action="store_true")
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
    args = parser.parse_args()

    if args.deterministic:
        args.churn_step_ratio = 0.0
        args.eta = 0.0

    if args.device is None:
        args.device = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    args.primary_device = args.device[0] if isinstance(args.device, list) else args.device
    args.use_multi_gpu = len(args.device) > 1 and all(d.startswith("cuda") for d in args.device)
    return args


def _load_pipeline(pretrained_path: str, primary_device: str, device_ids: list[int] | None = None) -> DBIMPipeline:
    path = Path(pretrained_path)
    if not path.is_dir():
        raise ValueError(
            f"Expected a checkpoint directory, got: {pretrained_path}. "
            "DBIM sample currently supports diffusers-style checkpoint folders only."
        )

    logger.info("Loading DBIM pipeline from checkpoint directory: %s", path)
    unet_subfolder = "ema_unet" if (path / "ema_unet").is_dir() else "unet"
    unet = DBIMUNet.from_pretrained(
        pretrained_path,
        subfolder=unet_subfolder,
        torch_dtype=torch.bfloat16,
    )

    scheduler_dir = path / "scheduler"
    scheduler_config = scheduler_dir / "scheduler_config.json"
    scheduler = None
    if scheduler_config.exists():
        scheduler = DBIMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    else:
        config_yaml = path / "config.yaml"
        if config_yaml.exists():
            with config_yaml.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            scheduler_kwargs = {
                "sigma_min": cfg.get("sigma_min", 0.002),
                "sigma_max": cfg.get("sigma_max", 1.0),
                "sigma_data": cfg.get("sigma_data", 0.5),
                "beta_d": cfg.get("beta_d", 2.0),
                "beta_min": cfg.get("beta_min", 0.1),
                "pred_mode": cfg.get("pred_mode", "vp"),
                "sampler": cfg.get("sampler", "dbim"),
                "eta": cfg.get("eta", 1.0),
                "order": cfg.get("order", 2),
                "lower_order_final": cfg.get("lower_order_final", True),
            }
            scheduler = DBIMScheduler(**scheduler_kwargs)
            logger.info(
                "Scheduler config missing under %s; reconstructed DBIMScheduler from %s",
                scheduler_dir,
                config_yaml,
            )
        else:
            scheduler = DBIMScheduler()
            logger.warning(
                "Scheduler config missing under %s and no config.yaml found. Using DBIMScheduler defaults.",
                scheduler_dir,
            )

    pipeline = DBIMPipeline(unet=unet, scheduler=scheduler).to(
        primary_device, dtype=torch.bfloat16
    )

    if device_ids is not None and len(device_ids) > 1:
        pipeline.unet = torch.nn.DataParallel(pipeline.unet, device_ids=device_ids)
        logger.info("Using DataParallel on GPUs %s", device_ids)
    return pipeline


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    submission_root = Path(args.submission_root)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = submission_root / args.task / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device_ids = None
    if args.use_multi_gpu:
        device_ids = [int(d.split(":")[1]) if ":" in d else i for i, d in enumerate(args.device)]
    pipeline = _load_pipeline(args.pretrained_model_name_or_path, args.primary_device, device_ids=device_ids)

    eval_resolution = args.resolution if args.resolution is not None else cfg.resolution
    # Use model_channels for source/target to match training (DDBM expects symmetric
    # channel counts in pixel space; 1-ch SAR is expanded to model_channels).
    dataset = MavicTDBIMDataset(
        task=args.task,
        split=args.split,
        resolution=eval_resolution,
        source_channels=cfg.model_channels,
        target_channels=cfg.model_channels,
        with_target=False,
    )

    if args.skip_existing:
        indices_to_process = [
            i for i in range(len(dataset))
            if not (output_dir / dataset.get_output_name(i)).exists()
        ]
        n_total = len(dataset)
        if not indices_to_process:
            logger.info("All %d outputs already exist in %s. Nothing to do.", n_total, output_dir)
            return
        if len(indices_to_process) < n_total:
            dataset = _SubsetWithOutputNames(dataset, indices_to_process)
            logger.info(
                "Skipping %d existing, processing %d remaining.",
                n_total - len(indices_to_process),
                len(indices_to_process),
            )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(
        "Generating samples for %s (%s), %d inputs, batch_size=%d, steps=%d, sampler=%s -> %s",
        args.task,
        args.split,
        len(dataset),
        args.batch_size,
        args.num_inference_steps,
        args.sampler,
        output_dir,
    )

    all_samples = []
    total_start = time.perf_counter()
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling"):
            source = batch[1].to(args.primary_device) * 2 - 1
            result = pipeline(
                source_image=source,
                num_inference_steps=args.num_inference_steps,
                sampler=args.sampler,
                guidance=args.guidance,
                cfg_scale=args.cfg_scale,
                churn_step_ratio=args.churn_step_ratio,
                eta=args.eta,
                order=args.order,
                lower_order_final=args.lower_order_final,
                clip_denoised=args.clip_denoised,
                output_type="pt",
            )
            images = result.images
            images_uint8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            images_uint8 = images_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for i, img_arr in enumerate(images_uint8):
                out_name = dataset.get_output_name(sample_idx + i)
                if img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(2)
                Image.fromarray(img_arr).save(output_dir / out_name)

            sample_idx += len(images_uint8)
            all_samples.append(images_uint8)

    total_elapsed = time.perf_counter() - total_start
    runtime_per_image = total_elapsed / len(dataset) if len(dataset) > 0 else 0.0
    use_gpu = args.primary_device.startswith("cuda")

    checkpoint_config = load_checkpoint_config(args.pretrained_model_name_or_path)
    extra_sampling = [
        f"Sampler: {args.sampler}",
        f"Num inference steps: {args.num_inference_steps}",
        f"Guidance scale: {args.guidance}",
        f"CFG scale: {args.cfg_scale}",
        f"Churn step ratio: {args.churn_step_ratio}",
        f"Eta (stochasticity): {args.eta}",
    ]
    if args.sampler == "dbim_high_order":
        extra_sampling.extend([f"Order: {args.order}", f"Lower order final: {args.lower_order_final}"])
    extra_sampling.append(f"Clip denoised: {args.clip_denoised}")
    detailed_description = build_detailed_description(
        model_name="DBIM",
        model_description="DBIM (Diffusion Bridge Implicit Model) - diffusion bridge implicit model for image-to-image translation. PyTorch implementation. UNet backbone: ADM-style (DiT-like) with concat conditioning.",
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

    if args.save_npz and all_samples:
        arr = np.concatenate(all_samples, axis=0)
        np.savez(output_dir / f"samples_{len(arr)}.npz", arr_0=arr)
        logger.info("Saved NPZ with %d samples.", len(arr))

    logger.info(
        "Sampling complete - %d images saved to %s (runtime: %.1fs total, %.2fs/image)",
        sample_idx,
        output_dir,
        total_elapsed,
        runtime_per_image,
    )


if __name__ == "__main__":
    main()
