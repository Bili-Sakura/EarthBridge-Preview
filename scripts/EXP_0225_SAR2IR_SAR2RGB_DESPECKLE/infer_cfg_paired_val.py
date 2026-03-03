#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Compare CFG scales on paired val set and save visual grids.

Each saved image is a single-row grid built with diffusers.make_image_grid:
    [input, target, pred(cfg1), pred(cfg2), ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import make_image_grid
from PIL import Image
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.dbim.config import rgb2ir_config, sar2eo_config, sar2ir_config, sar2rgb_config
from examples.ddbm.dataset_wrapper import PairedValDataset
from examples.eval_common import resolve_manifest
from src.models.unet_dbim import DBIMUNet
from src.pipelines.dbim import DBIMPipeline

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def _parse_cfg_scales(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("cfg_scales cannot be empty")
    return vals


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (C,H,W) tensor in [0,1] or [-1,1] to PIL RGB."""
    t = t.detach().cpu()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.min() < 0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0, 1)
    arr = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return Image.fromarray(arr).convert("RGB")


def _load_pipeline(checkpoint_dir: Path, device: str) -> DBIMPipeline:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    pipeline = DBIMPipeline.from_pretrained(str(checkpoint_dir))
    ema_dir = checkpoint_dir / "ema_unet"
    if ema_dir.is_dir():
        logger.info("Using EMA UNet from %s", ema_dir)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        pipeline.unet = DBIMUNet.from_pretrained(
            str(checkpoint_dir),
            subfolder="ema_unet",
            torch_dtype=dtype,
        )
    pipeline = pipeline.to(device)
    pipeline.unet.eval()
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DBIM CFG grid inference on paired val set")
    parser.add_argument("--task", type=str, default="sar2ir", choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Diffusers-style checkpoint directory")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Paired val manifest. Defaults to task config paired_val_manifest.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save CFG grids")
    parser.add_argument("--cfg_scales", type=str, default="1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, default="dbim", choices=["dbim", "dbim_high_order", "heun"])
    parser.add_argument("--churn_step_ratio", type=float, default=0.33)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--order", type=int, default=2, choices=[2, 3])
    parser.add_argument("--lower_order_final", action="store_true", default=True)
    parser.add_argument("--clip_denoised", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_scales = _parse_cfg_scales(args.cfg_scales)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task_cfg = _TASK_CONFIG_MAP[args.task]()
    manifest_raw = args.manifest or task_cfg.paired_val_manifest
    if not manifest_raw:
        raise ValueError(f"No paired_val manifest configured for task={args.task}")
    manifest_path = resolve_manifest(manifest_raw)

    output_dir = Path(
        args.output_dir
        or (_PROJECT_ROOT / f"ckpt/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/cfg_grids/{args.task}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = _load_pipeline(Path(args.checkpoint_dir), device)

    eval_resolution = getattr(task_cfg, "output_resolution", None) or task_cfg.resolution
    src_ch = task_cfg.model_channels
    tgt_ch = task_cfg.model_channels
    dataset = PairedValDataset(
        manifest_path=manifest_path,
        resolution=eval_resolution,
        source_channels=src_ch,
        target_channels=tgt_ch,
        use_sar_despeckle=getattr(task_cfg, "use_sar_despeckle", False),
        sar_despeckle_kernel_size=getattr(task_cfg, "sar_despeckle_kernel_size", 5),
        sar_despeckle_strength=getattr(task_cfg, "sar_despeckle_strength", 0.6),
    )

    n = min(len(dataset), max(0, args.max_samples))
    logger.info(
        "CFG compare task=%s pairs=%d/%d cfg_scales=%s manifest=%s",
        args.task,
        n,
        len(dataset),
        cfg_scales,
        manifest_path,
    )

    with torch.no_grad():
        for i in tqdm(range(n), desc=f"CFG grid {args.task}", unit="img"):
            target, source = dataset[i]
            source_inp = source.unsqueeze(0).to(device) * 2 - 1

            tiles = [_tensor_to_pil(source), _tensor_to_pil(target)]
            for cfg_scale in cfg_scales:
                result = pipeline(
                    source_image=source_inp,
                    num_inference_steps=args.num_inference_steps,
                    sampler=args.sampler,
                    guidance=args.guidance,
                    cfg_scale=cfg_scale,
                    churn_step_ratio=args.churn_step_ratio,
                    eta=args.eta,
                    order=args.order,
                    lower_order_final=args.lower_order_final,
                    clip_denoised=args.clip_denoised,
                    output_type="pt",
                )
                pred = result.images[0]
                tiles.append(_tensor_to_pil(pred))

            grid = make_image_grid(tiles, rows=1, cols=len(tiles))
            src_stem = Path(dataset._pairs[i][0]).stem
            grid.save(output_dir / f"cfg_grid_{args.task}_{i:03d}_{src_stem}.png")

    logger.info("Saved %d CFG grids to %s", n, output_dir)


if __name__ == "__main__":
    main()
