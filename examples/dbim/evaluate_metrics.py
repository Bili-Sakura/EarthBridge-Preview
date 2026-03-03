#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Manual MAVIC-T metric evaluation for a trained DBIM checkpoint on a paired validation set.

Same interface as examples/cut/evaluate_metrics.py. Normalization follows
official-docs/evaluation.md.

Usage::

    conda activate rsgen
    python -m examples.dbim.evaluate_metrics \
        --checkpoint_dir ./ckpt/dbim/sar2eo/checkpoint-10000 \
        --manifest datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2eo.txt \
        --task sar2eo \
        --batch_size 8 \
        --num_inference_steps 40

    # MultiDiffusion-style (resize source to 1024, tiled 512px windows):
    python -m examples.dbim.evaluate_metrics ... --resolution 1024 --output_size 1024 1024 --view_batch_size 4
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.eval_common import resolve_manifest, run_metric_evaluation
from examples.dbim.config import TaskConfig, sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAVIC-T metrics (DBIM) on a paired val set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to diffusers-style checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to paired val manifest (source\\ttarget per line).")
    parser.add_argument("--task", type=str, default="sar2eo", choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Classifier-Free Guidance scale. Defaults to checkpoint config or 1.0.",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_fid", action="store_true", help="Disable FID (faster).")
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="MultiDiffusion: resize source to (H,W) then tiled 512px windows (e.g. 1024 1024).",
    )
    parser.add_argument(
        "--view_batch_size",
        type=int,
        default=1,
        help="MultiDiffusion: batch views for speed (e.g. 4).",
    )
    parser.add_argument(
        "--multidiffusion_input_size",
        type=int,
        default=512,
        help="When output_size is set: resize val source to this size before pipeline (val is 1024px). Default 512.",
    )
    parser.add_argument(
        "--multidiffusion_window_size",
        type=int,
        default=None,
        help="MultiDiffusion: tile window size in pixels (default 512). Overrides src.utils.multidiffusion default.",
    )
    parser.add_argument(
        "--multidiffusion_stride",
        type=int,
        default=None,
        help="MultiDiffusion: stride between tiles in pixels (default 64). Overrides src.utils.multidiffusion default.",
    )
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def load_config(checkpoint_dir: Path, task: str) -> TaskConfig:
    cfg = _TASK_CONFIG_MAP[task]()
    config_yaml = checkpoint_dir / "config.yaml"
    if config_yaml.is_file():
        import yaml
        with open(config_yaml) as f:
            saved = yaml.safe_load(f)
        if saved:
            for key in (
                "resolution",
                "source_channels",
                "target_channels",
                "model_channels",
                "use_latent_target",
                "validation_resolution",
                "num_inference_steps",
                "guidance",
                "cfg_scale",
                "churn_step_ratio",
            ):
                if key in saved and saved[key] is not None:
                    setattr(cfg, key, saved[key])
    return cfg


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    manifest_path = resolve_manifest(args.manifest)
    cfg = load_config(checkpoint_dir, args.task)
    resolution = args.resolution or getattr(cfg, "validation_resolution", None) or cfg.resolution

    from src.pipelines.dbim import DBIMPipeline
    from src.models.unet_dbim import DBIMUNet

    logger.info("Loading DBIM pipeline from %s", checkpoint_dir)
    pipeline = DBIMPipeline.from_pretrained(str(checkpoint_dir))
    ema_dir = checkpoint_dir / "ema_unet"
    if ema_dir.is_dir():
        logger.info("Loading EMA UNet from %s", ema_dir)
        pipeline.unet = DBIMUNet.from_pretrained(str(checkpoint_dir), subfolder="ema_unet")
    pipeline = pipeline.to(args.device)
    pipeline.unet.eval()
    unet_device = next(pipeline.unet.parameters()).device
    logger.info("Using device: %s (CUDA available: %s)", unet_device, torch.cuda.is_available())

    # CLI takes precedence over checkpoint config
    num_steps = args.num_inference_steps
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else getattr(cfg, "cfg_scale", 1.0)

    output_size = tuple(args.output_size) if args.output_size else None
    # When using MultiDiffusion, val set is 1024px; resize source to 512px so pipeline sees 512→1024
    md_input_size = args.multidiffusion_input_size

    def inference_fn(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_inp = source * 2 - 1
        if output_size is not None:
            # Resize 1024px val source to 512px; pipeline will upsample to output_size with tiling
            source_inp = F.interpolate(
                source_inp, size=(md_input_size, md_input_size), mode="bilinear", align_corners=False
            )
        result = pipeline(
            source_image=source_inp,
            num_inference_steps=num_steps,
            sampler=getattr(cfg, "sampler", "dbim"),
            guidance=getattr(cfg, "guidance", 1.0),
            cfg_scale=cfg_scale,
            churn_step_ratio=getattr(cfg, "churn_step_ratio", 0.33),
            eta=getattr(cfg, "eta", 1.0),
            order=getattr(cfg, "order", 2),
            lower_order_final=getattr(cfg, "lower_order_final", True),
            clip_denoised=getattr(cfg, "clip_denoised", False),
            output_type="pt",
            output_size=output_size,
            view_batch_size=args.view_batch_size,
            multidiffusion_window_size=args.multidiffusion_window_size,
            multidiffusion_stride=args.multidiffusion_stride,
        )
        return (result.images + 1) * 0.5

    # In pixel space, DBIM expects symmetric channel counts; use model_channels for source.
    src_ch = cfg.model_channels if not getattr(cfg, "use_latent_target", False) else cfg.source_channels
    try:
        run_metric_evaluation(
            manifest_path=manifest_path,
            resolution=resolution,
            source_channels=src_ch,
            target_channels=cfg.target_channels,
            device=args.device,
            batch_size=args.batch_size,
            no_fid=args.no_fid,
            inference_fn=inference_fn,
        )
    finally:
        # Release GPU memory so next task (or other processes) can use it
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
