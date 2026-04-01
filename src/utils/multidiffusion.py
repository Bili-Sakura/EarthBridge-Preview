# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Sliding-window view layout for tiled (MultiDiffusion-style) inference.

Overlapping crops are listed as (h_start, h_end, w_start, w_end) in the same
grid units as height/width (pixels for pixel pipelines, latent cells for latent).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Pixel-space tiling defaults (512px windows, 64px stride).
DEFAULT_PIXEL_WINDOW_SIZE = 512
DEFAULT_PIXEL_STRIDE = 64

# Latent-space tiling defaults (e.g. 64x64 latents for 512px at 8x VAE scale).
DEFAULT_LATENT_WINDOW_SIZE = 64
DEFAULT_LATENT_STRIDE = 32


def get_views(
    height: int,
    width: int,
    window_size: int,
    stride: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """Enumerate overlapping windows that cover a 2D grid.

    Parameters
    ----------
    height, width
        Grid extent (pixels or latent indices).
    window_size
        Square window side length; clamped to ``min(window_size, height, width)``.
    stride
        Step between window starts; defaults to ``DEFAULT_LATENT_STRIDE`` when omitted
        (for backward compatibility with three-argument call sites).
    """
    if height < 1 or width < 1:
        return []
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if stride is None:
        stride = DEFAULT_LATENT_STRIDE
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    win = min(window_size, height, width)

    def _axis_starts(span: int, w: int, s: int) -> List[int]:
        if span <= w:
            return [0]
        starts = []
        pos = 0
        while True:
            starts.append(pos)
            if pos + w >= span:
                break
            next_pos = pos + s
            # Snap last window to the far edge so the grid is fully covered.
            if next_pos + w > span:
                starts.append(span - w)
                break
            pos = next_pos
        # Deduplicate while preserving order
        out: List[int] = []
        seen = set()
        for p in starts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    h_starts = _axis_starts(height, win, stride)
    w_starts = _axis_starts(width, win, stride)
    views: List[Tuple[int, int, int, int]] = []
    for hs in h_starts:
        he = hs + win
        for ws in w_starts:
            we = ws + win
            views.append((hs, he, ws, we))
    return views
