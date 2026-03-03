# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""MAVIC-T official evaluation metrics.

This module implements the three evaluation metrics used by the 4th MAVIC-T
challenge (https://www.codabench.org/competitions/12566/) together with the
composite task-score and overall-score formulas. Normalization follows the
official evaluation statement (see ``official-docs/evaluation.md``): all
three components are scaled to [0, 1] so the task score is their average.

Metrics
-------
- **LPIPS** - Learned Perceptual Image Patch Similarity (VGG-16); output scaled for normalization.
- **FID**  - Fréchet Inception Distance (InceptionV3); normalized via 2/π · arctan(FID) in the score.
- **L1**   - Mean pixel-wise absolute difference; pixel values in [0, 1] so L1 is in [0, 1].

Scoring
-------
``task_score = (2/π · arctan(FID)  +  LPIPS  +  L1) / 3``

``overall_score = mean(task_scores)`` with a penalty of **1** added for each
unattempted domain (out of the four: sar2eo, sar2rgb, sar2ir, rgb2ir).

Training loss
-------------
:class:`MavicCriterion` provides a differentiable combination of LPIPS and L1
so that the model can be directly optimised toward the evaluation metric.
FID is distribution-level and cannot be used as a per-sample loss, so it is
excluded from the training criterion but included in evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LPIPS (torchmetrics exposes it under torchmetrics.image.lpip)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# FID (optional - requires torchvision for InceptionV3 weights)
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except (ImportError, RuntimeError):
    FID_AVAILABLE = False
    FrechetInceptionDistance = None


# ---------------------------------------------------------------------------
# L1 metric
# ---------------------------------------------------------------------------

def compute_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute pixel-wise error.

    Parameters
    ----------
    predictions, targets : torch.Tensor
        Image tensors in ``[0, 1]`` range with shape ``(N, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Scalar L1 value.
    """
    return F.l1_loss(predictions, targets)


# ---------------------------------------------------------------------------
# LPIPS metric (wraps torchmetrics)
# ---------------------------------------------------------------------------

class LPIPS(nn.Module):
    """VGG-16 based Learned Perceptual Image Patch Similarity.

    Thin wrapper around ``torchmetrics.image.LearnedPerceptualImagePatchSimilarity``
    that accepts images in ``[0, 1]`` and re-scales them to ``[-1, 1]`` as
    expected by the underlying network.

    Parameters
    ----------
    net_type : str
        Backbone network (``"vgg"`` for the official MAVIC-T evaluation).
    """

    def __init__(self, net_type: str = "vgg") -> None:
        super().__init__()
        self._lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPIPS.

        Parameters
        ----------
        predictions, targets : torch.Tensor
            Image tensors in ``[0, 1]`` range with shape ``(N, C, H, W)``.
            For grayscale (C=1), channels are replicated to 3.

        Returns
        -------
        torch.Tensor
            Scalar LPIPS value (lower is better).
        """
        # LPIPS expects [-1, 1] and 3-channel images
        preds = predictions * 2 - 1
        tgts = targets * 2 - 1
        if preds.shape[1] == 1:
            preds = preds.expand(-1, 3, -1, -1)
            tgts = tgts.expand(-1, 3, -1, -1)
        return self._lpips(preds, tgts)


# ---------------------------------------------------------------------------
# Task score & overall score
# ---------------------------------------------------------------------------

def task_score(fid: float, lpips: float, l1: float) -> float:
    """Compute the MAVIC-T task score.

    ``score = (2/π · arctan(FID) + LPIPS + L1) / 3``

    Lower is better (all three components are distances).
    """
    normalised_fid = (2.0 / math.pi) * math.atan(fid)
    return (normalised_fid + lpips + l1) / 3.0


def overall_score(
    task_scores: Dict[str, float],
    all_tasks: Sequence[str] = ("sar2eo", "sar2rgb", "sar2ir", "rgb2ir"),
) -> float:
    """Compute the MAVIC-T overall score.

    ``overall = mean(task_scores) + penalty``

    A penalty of **1** is added for each unattempted task.
    Lower is better.
    """
    attempted = [task_scores[t] for t in all_tasks if t in task_scores]
    unattempted = sum(1 for t in all_tasks if t not in task_scores)

    if not attempted:
        return float(len(all_tasks))

    return sum(attempted) / len(all_tasks) + unattempted


# ---------------------------------------------------------------------------
# MetricResults dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricResults:
    """Container for MAVIC-T evaluation results."""

    lpips: float
    fid: Optional[float] = None
    l1: float = 0.0

    @property
    def score(self) -> Optional[float]:
        """MAVIC-T task score, or *None* if FID is unavailable."""
        if self.fid is None:
            return None
        return task_score(self.fid, self.lpips, self.l1)

    def __repr__(self) -> str:
        parts = [f"LPIPS: {self.lpips:.4f}", f"L1: {self.l1:.4f}"]
        if self.fid is not None:
            parts.append(f"FID: {self.fid:.2f}")
        s = self.score
        if s is not None:
            parts.append(f"TaskScore: {s:.4f}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {"lpips": self.lpips, "fid": self.fid, "l1": self.l1, "score": self.score}


# ---------------------------------------------------------------------------
# MetricCalculator - batch-wise accumulation using torchmetrics
# ---------------------------------------------------------------------------

class MetricCalculator:
    """Accumulates MAVIC-T metrics (LPIPS, FID, L1) over batches.

    Uses ``torchmetrics`` for LPIPS and FID so there is no need for
    manual InceptionV3 feature extraction or scipy-based Fréchet distance.

    Parameters
    ----------
    device : str
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    compute_fid : bool
        Whether to compute FID. Requires ``torchmetrics[image]``.
    net_type : str
        LPIPS backbone (``"vgg"`` for the official MAVIC-T evaluation).
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_fid: bool = True,
        net_type: str = "vgg",
    ) -> None:
        self.device = device
        self._compute_fid = compute_fid and FID_AVAILABLE

        # LPIPS - torchmetrics (expects [-1, 1])
        self._lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device)

        # FID - torchmetrics (expects uint8 [0, 255])
        self._fid: Optional[FrechetInceptionDistance] = None
        if self._compute_fid and FrechetInceptionDistance is not None:
            try:
                self._fid = FrechetInceptionDistance(normalize=False).to(device)
            except Exception:
                self._compute_fid = False

        self._l1_total: float = 0.0
        self._num_samples: int = 0
        self._num_pixels: int = 0  # for L1: mean over pixels so value is in [0, 1] (official normalization)

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._l1_total = 0.0
        self._num_samples = 0
        self._num_pixels = 0
        self._lpips.reset()
        if self._fid is not None:
            self._fid.reset()

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Feed a batch of images in ``[0, 1]`` range (N, C, H, W).

        Parameters
        ----------
        predictions, targets : torch.Tensor
            Float tensors in ``[0, 1]`` with shape ``(N, C, H, W)``.
        """
        predictions = predictions.clamp(0, 1)
        targets = targets.clamp(0, 1)

        # L1 - mean absolute pixel error in [0, 1] (official: "Pixel values adjusted to fit within the desired range")
        batch_size = predictions.shape[0]
        self._l1_total += F.l1_loss(predictions, targets, reduction="sum").item()
        self._num_samples += batch_size
        self._num_pixels += predictions.numel()

        # LPIPS (expects [-1, 1], 3 channels) - accumulates internally
        preds_lp = predictions * 2 - 1
        tgts_lp = targets * 2 - 1
        if preds_lp.shape[1] == 1:
            preds_lp = preds_lp.expand(-1, 3, -1, -1)
            tgts_lp = tgts_lp.expand(-1, 3, -1, -1)
        self._lpips.update(preds_lp, tgts_lp)

        # FID (expects uint8, 3 channels)
        if self._fid is not None:
            preds_uint8 = (predictions * 255).to(torch.uint8)
            tgts_uint8 = (targets * 255).to(torch.uint8)
            if preds_uint8.shape[1] == 1:
                preds_uint8 = preds_uint8.expand(-1, 3, -1, -1)
                tgts_uint8 = tgts_uint8.expand(-1, 3, -1, -1)
            self._fid.update(tgts_uint8, real=True)
            self._fid.update(preds_uint8, real=False)

    def compute(self) -> MetricResults:
        """Return aggregated :class:`MetricResults`."""
        if self._num_samples == 0:
            return MetricResults(lpips=0.0, l1=0.0, fid=None)

        # L1: mean over all pixels so score is in [0, 1] per official normalization
        l1_val = float(self._l1_total / self._num_pixels) if self._num_pixels > 0 else 0.0

        fid_val: Optional[float] = None
        if self._fid is not None:
            try:
                fid_val = self._fid.compute().item()
            except Exception:
                fid_val = None

        return MetricResults(
            lpips=self._lpips.compute().item(),
            l1=l1_val,
            fid=fid_val,
        )


# ---------------------------------------------------------------------------
# Differentiable training criterion (LPIPS + L1)
# ---------------------------------------------------------------------------

class MavicCriterion(nn.Module):
    """Differentiable training loss matching the MAVIC-T evaluation metric.

    The official evaluation combines LPIPS, FID, and L1.  FID is a
    distribution-level metric and cannot be used per-sample, so this criterion
    uses only the two sample-level components:

    ``loss = lpips_weight · LPIPS(pred, target) + l1_weight · L1(pred, target)``

    Parameters
    ----------
    lpips_weight : float
        Weight for the LPIPS term (default ``1.0``).
    l1_weight : float
        Weight for the L1 term (default ``1.0``).
    net_type : str
        LPIPS backbone (default ``"vgg"``).
    """

    def __init__(
        self,
        lpips_weight: float = 1.0,
        l1_weight: float = 1.0,
        net_type: str = "vgg",
    ) -> None:
        super().__init__()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self._lpips = LPIPS(net_type=net_type)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        predictions, targets : torch.Tensor
            ``(N, C, H, W)`` tensors in ``[0, 1]`` range.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        l1 = compute_l1(predictions, targets)
        lpips_val = self._lpips(predictions, targets)
        return self.lpips_weight * lpips_val + self.l1_weight * l1
