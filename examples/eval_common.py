# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Shared logic for manual MAVIC-T metric evaluation across baselines.

Used by examples/*/evaluate_metrics.py. Builds PairedValDataset, runs an
inference callable per batch, accumulates LPIPS/FID/L1, and prints the
task score. Normalization follows official-docs/evaluation.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.utils.metrics import MetricResults

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def resolve_manifest(manifest_arg: str) -> Path:
    """Resolve manifest path (cwd or project root)."""
    from examples.ddbm.dataset_wrapper import resolve_paired_val_manifest

    resolved = resolve_paired_val_manifest(manifest_arg)
    if resolved is not None:
        return resolved
    p = Path(manifest_arg)
    if p.is_file():
        return p
    candidate = _PROJECT_ROOT / manifest_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Manifest not found: {manifest_arg} (tried cwd and project root)")


def run_metric_evaluation(
    manifest_path: Path,
    resolution: int,
    source_channels: int,
    target_channels: int,
    device: str,
    batch_size: int,
    no_fid: bool,
    inference_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> MetricResults:
    """Run MAVIC-T metrics over a paired val set.

    inference_fn(source_batch, target_batch) must return predictions in [0, 1]
    on the same device, shape (N, C, H, W) compatible with target for metrics.
    """
    from examples.ddbm.dataset_wrapper import PairedValDataset
    from src.utils.metrics import MetricCalculator, MetricResults

    val_ds = PairedValDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        source_channels=source_channels,
        target_channels=target_channels,
        return_order="source_target",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info("Paired val set: %s (%d pairs), resolution=%d", manifest_path, len(val_ds), resolution)

    metric_calc = MetricCalculator(device=device, compute_fid=not no_fid, net_type="vgg")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            source, target = batch
            source = source.to(device)
            target = target.to(device)
            pred = inference_fn(source, target)
            pred = pred.clamp(0, 1)

            # Align channels for metrics (e.g. 1ch target vs 3ch pred)
            if target_channels == 1 and pred.shape[1] == 3:
                pred = pred.mean(dim=1, keepdim=True)
            elif pred.shape[1] != target.shape[1]:
                if pred.shape[1] == 3 and target.shape[1] == 1:
                    pred = pred.mean(dim=1, keepdim=True)
                elif pred.shape[1] == 1 and target.shape[1] == 3:
                    pred = pred.repeat(1, 3, 1, 1)
            metric_calc.update(pred, target)

    results = metric_calc.compute()
    logger.info("Metrics: %s", results)
    if results.score is not None:
        logger.info("Task score (2/π·arctan(FID)+LPIPS+L1)/3 = %.4f", results.score)
    else:
        logger.info("Task score: (FID not computed; use without --no_fid for full score)")
    print(
        f"lpips={results.lpips:.4f} l1={results.l1:.4f}"
        + (f" fid={results.fid:.2f}" if results.fid is not None else " fid=N/A")
        + (f" task_score={results.score:.4f}" if results.score is not None else "")
    )
    return results
