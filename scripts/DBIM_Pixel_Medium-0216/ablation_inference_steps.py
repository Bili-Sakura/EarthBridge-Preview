#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Ablation: inference steps 1, 10, 100, 1000 on paired_val set for each task.

For each sample: grid row = [input, ground_truth, pred_1, pred_10, pred_100, pred_1000].
Runs on all four tasks (sar2ir, sar2eo, sar2rgb, rgb2ir) when invoked by run_ablation_steps.sh.

Usage:
    TASK=sar2ir CKPT_PATH=/path MANIFEST=/path/paired_val_sar2ir.txt python ablation_inference_steps.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.unet_dbim import DBIMUNet
from src.pipelines.dbim import DBIMPipeline
from src.schedulers import DBIMScheduler

from examples.dbim.config import sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config
from examples.ddbm.dataset_wrapper import PairedValDataset
from examples.eval_common import resolve_manifest

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TASK = os.environ.get("TASK", "sar2ir")
CKPT_PATH = os.environ.get("CKPT_PATH", "")
MANIFEST = os.environ.get("MANIFEST", "")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("/data/projects/4th-MAVIC-T/temp")
STEP_VALUES = [1, 10, 100, 1000]
SEED = 42

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def _load_pipeline(pretrained_path: str, device: str) -> DBIMPipeline:
    path = Path(pretrained_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    logger.info("Loading pipeline from %s", path)
    unet_subfolder = "ema_unet" if (path / "ema_unet").is_dir() else "unet"
    unet = DBIMUNet.from_pretrained(
        pretrained_path,
        subfolder=unet_subfolder,
        torch_dtype=torch.bfloat16,
    )

    scheduler_config = path / "scheduler" / "scheduler_config.json"
    if scheduler_config.exists():
        scheduler = DBIMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    else:
        config_yaml = path / "config.yaml"
        if config_yaml.exists():
            with config_yaml.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            scheduler = DBIMScheduler(
                sigma_min=cfg.get("sigma_min", 0.002),
                sigma_max=cfg.get("sigma_max", 1.0),
                sigma_data=cfg.get("sigma_data", 0.5),
                beta_d=cfg.get("beta_d", 2.0),
                beta_min=cfg.get("beta_min", 0.1),
                pred_mode=cfg.get("pred_mode", "vp"),
                sampler=cfg.get("sampler", "dbim"),
                eta=cfg.get("eta", 1.0),
                order=cfg.get("order", 2),
                lower_order_final=cfg.get("lower_order_final", True),
            )
        else:
            scheduler = DBIMScheduler()

    return DBIMPipeline(unet=unet, scheduler=scheduler).to(
        device, dtype=torch.bfloat16
    )


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (C,H,W) tensor in [0,1] or [-1,1] to PIL Image."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    t = t.cpu()
    if t.min() >= 0:
        arr = (t * 255).clamp(0, 255).to(torch.uint8)
    else:
        arr = ((t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    arr = arr.permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return Image.fromarray(arr)


def main():
    if not CKPT_PATH or not MANIFEST:
        raise ValueError("CKPT_PATH and MANIFEST must be set (via env or run_ablation_steps.sh)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg_fn = _TASK_CONFIG_MAP.get(TASK)
    if cfg_fn is None:
        raise ValueError(f"Unknown task: {TASK}")
    cfg = cfg_fn()
    resolution = getattr(cfg, "validation_resolution", None) or cfg.resolution

    manifest_path = resolve_manifest(MANIFEST)
    # In pixel space, DBIM expects symmetric channel counts; use model_channels for source.
    src_ch = cfg.model_channels if not getattr(cfg, "use_latent_target", False) else cfg.source_channels
    dataset = PairedValDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        source_channels=src_ch,
        target_channels=cfg.target_channels,
        return_order="target_source",
    )

    device = DEVICE
    pipeline = _load_pipeline(CKPT_PATH, device)

    logger.info(
        "Ablation: task=%s, steps=%s, samples=%d",
        TASK,
        STEP_VALUES,
        len(dataset),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"{TASK} samples", unit="sample"):
            target, source = dataset[idx]
            source_batch = source.unsqueeze(0).to(device) * 2 - 1  # [0,1] -> [-1,1]

            row_images = [
                _tensor_to_pil(source),   # input
                _tensor_to_pil(target),   # ground truth
            ]

            for n_steps in STEP_VALUES:
                result = pipeline(
                    source_image=source_batch,
                    num_inference_steps=n_steps,
                    sampler="dbim",
                    guidance=1.0,
                    churn_step_ratio=0.33,
                    eta=1.0,
                    order=2,
                    lower_order_final=True,
                    clip_denoised=False,
                    output_type="pt",
                )
                img = result.images[0]
                row_images.append(_tensor_to_pil(img))

            # One row: input | gt | pred_1 | pred_10 | pred_100 | pred_1000
            grid = make_image_grid(row_images, rows=1, cols=len(row_images))
            stem = Path(dataset._pairs[idx][0]).stem
            out_path = OUTPUT_DIR / f"ablation_steps_{TASK}_{stem}.png"
            grid.save(out_path)

    logger.info("Saved %d grids to %s", len(dataset), OUTPUT_DIR)


if __name__ == "__main__":
    main()
