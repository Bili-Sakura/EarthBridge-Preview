# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""PyTorch Dataset wrapper around ``MavicTImageToImageDataset`` for CUT.

This module bridges the HuggingFace ``datasets.Dataset`` returned by
:class:`src.utils.mavic_t_dataset.MavicTImageToImageDataset` and the PyTorch
:class:`torch.utils.data.Dataset` interface expected by CUT training.

It handles:
* Reading images lazily from disk (via the path columns).
* Random crop (training): direct crop from original image at native scale—no resize.
  Optional: when load_size > resolution, resize to load_size then random crop
  (resize-and-crop mode; not used in our experiments).
* Resize (val/test or when image smaller than resolution): resize to resolution.
* Channel adaptation (expanding/repeating channels to ``model_channels``).
* Normalising pixel values to [0, 1] as float32 tensors.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

from typing import Set

# Ensure the project root is importable so we can import from ``src``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.mavic_t_dataset import MavicTImageToImageDataset  # noqa: E402
from examples.ddbm.dataset_wrapper import (  # noqa: E402
    _despeckle_tensor,
    _load_paired_val_exclude_set,
    _load_sar2rgb_sup_records,
    resolve_sar2rgb_sup_manifest,
)


def _load_image_as_tensor(
    path: str,
    channels: int,
    resolution: int,
    crop_pos: Optional[Tuple[int, int]] = None,
    use_random_crop: bool = False,
    load_size: Optional[int] = None,
) -> torch.Tensor:
    """Load an image from *path*, return ``(C, H, W)`` float32 in [0, 1].

    Modes:
    - use_random_crop + crop_pos, load_size=None: direct crop from original (preserves scale).
    - use_random_crop + crop_pos, load_size>resolution: resize to load_size, then crop.
    - load_size set and load_size<resolution: resize to load_size, then resize to resolution.
    - otherwise: resize to resolution.
    """
    img = Image.open(path)
    w, h = img.size

    if load_size is not None and load_size != resolution:
        # Resize mode: always resize to load_size first (works for both > and < resolution).
        img = img.resize((load_size, load_size), Image.BILINEAR)
        if load_size > resolution and use_random_crop and crop_pos is not None:
            x, y = crop_pos
            img = img.crop((x, y, x + resolution, y + resolution))
        else:
            img = img.resize((resolution, resolution), Image.BILINEAR)
    elif use_random_crop and crop_pos is not None and w >= resolution and h >= resolution:
        # Direct crop from original (no resize).
        x, y = crop_pos
        img = img.crop((x, y, x + resolution, y + resolution))
    else:
        img = img.resize((resolution, resolution), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)

    # Normalise to [0, 1]
    if arr.max() > 1.0:
        if arr.dtype == np.float32 and arr.max() > 255.0:
            arr = arr / 65535.0  # uint16 TIFF
        else:
            arr = arr / 255.0

    # Ensure 3-D: (H, W, C)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Channel adaptation
    c = arr.shape[2]
    if c < channels:
        arr = np.repeat(arr, channels // c + 1, axis=2)[:, :, :channels]
    elif c > channels:
        arr = arr[:, :, :channels]

    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
    return tensor


def _load_exclude_set(exclude_file: Optional[str]) -> Set[str]:
    """Load a set of absolute paths to exclude from training."""
    if not exclude_file:
        return set()
    path = Path(exclude_file)
    if not path.is_file():
        return set()
    out: Set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(str(Path(line).resolve()))
    return out


class MavicTCUTDataset(Dataset):
    """A PyTorch :class:`Dataset` that loads MAVIC-T image pairs for CUT.

    Each ``__getitem__`` returns ``(source_tensor, target_tensor)`` following the
    CUT convention where ``real_A = source`` and ``real_B = target``.  Both
    tensors are in ``[0, 1]`` range with shape
    ``(model_channels, resolution, resolution)``.

    Parameters
    ----------
    task : str
        One of ``sar2eo``, ``rgb2ir``, ``sar2ir``, ``sar2rgb``.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    resolution : int
        Spatial resolution (crop size) of output images.
    load_size : int or None
        If set (training only): resize to load_size first. Then: if load_size>resolution,
        random crop to resolution; if load_size<resolution, resize to resolution.
        Not used in our experiments.
    model_channels : int
        Number of channels the model operates in.
    with_target : bool or None
        Whether to load target images.  Defaults to ``True`` for train, ``False``
        for val/test.
    use_augmented : bool
        If ``True`` and ``split == "train"``, also include the ``*_crop_aug``
        variant as additional samples.
    use_random_crop : bool
        If ``True`` and ``split == "train"``, apply random crop to ``resolution``.
    use_horizontal_flip : bool
        If ``True`` and ``split == "train"``, randomly apply horizontal flip
        (applied consistently to both source and target).
    use_vertical_flip : bool
        If ``True`` and ``split == "train"``, randomly apply vertical flip
        (applied consistently to both source and target).
    refined_root, eval_root : str or Path or None
        Forwarded to :class:`MavicTImageToImageDataset`.
    paired_val_manifest : str or None
        Path to paired_val_<task>.txt. When loading train, paths in this manifest
        are excluded so the train set does not overlap with the golden val set.
    sar2rgb_sup_manifest : str or None
        Path to paired_sar2rgb_sup.txt. When task is sar2rgb and split is train,
        these additional supervised SAR→RGB pairs are appended to the training set.
    """

    def __init__(
        self,
        task: str,
        split: str = "train",
        resolution: int = 512,
        load_size: Optional[int] = None,
        source_channels: Optional[int] = None,
        target_channels: Optional[int] = None,
        model_channels: int = 3,
        with_target: Optional[bool] = None,
        use_augmented: bool = False,
        use_random_crop: bool = True,
        use_horizontal_flip: bool = False,
        use_vertical_flip: bool = False,
        refined_root: Optional[str] = None,
        eval_root: Optional[str] = None,
        exclude_file: Optional[str] = None,
        paired_val_manifest: Optional[str] = None,
        sar2rgb_sup_manifest: Optional[str] = None,
        use_sar_despeckle: bool = False,
        sar_despeckle_kernel_size: int = 5,
        sar_despeckle_strength: float = 0.6,
    ) -> None:
        if use_random_crop and split == "train" and (resolution is None or resolution <= 0):
            raise ValueError(
                "When use_random_crop is True for train split, resolution must be set and > 0. "
                f"Got resolution={resolution}."
            )
        super().__init__()
        self.task = task
        self.split = split
        self.resolution = resolution
        self.load_size = load_size
        self.source_channels = source_channels or model_channels
        self.target_channels = target_channels or model_channels
        self.use_random_crop = use_random_crop and split == "train"
        self.use_horizontal_flip = use_horizontal_flip and split == "train"
        self.use_vertical_flip = use_vertical_flip and split == "train"
        self.use_sar_despeckle = use_sar_despeckle and task.startswith("sar2")
        self.sar_despeckle_kernel_size = max(1, int(sar_despeckle_kernel_size))
        self.sar_despeckle_strength = float(max(0.0, min(1.0, sar_despeckle_strength)))

        if with_target is None:
            with_target = split == "train"
        self.with_target = with_target

        kwargs = {}
        if refined_root is not None:
            kwargs["refined_root"] = refined_root
        if eval_root is not None:
            kwargs["eval_root"] = eval_root
        loader = MavicTImageToImageDataset(**kwargs)

        # Load the base dataset
        ds = loader.load(split=split, task=task, with_target=with_target, load_images=False)
        self._records = list(ds)

        # Optionally add augmented samples for training
        if use_augmented and split == "train" and not task.endswith("_crop_aug"):
            aug_task = f"{task}_crop_aug"
            try:
                ds_aug = loader.load(split="train", task=aug_task, with_target=with_target, load_images=False)
                self._records.extend(list(ds_aug))
            except (ValueError, FileNotFoundError):
                pass  # augmented variant may not exist for all tasks

        if task == "sar2rgb" and split == "train" and sar2rgb_sup_manifest:
            resolved_sup = resolve_sar2rgb_sup_manifest(sar2rgb_sup_manifest)
            if resolved_sup is not None:
                sup_records = _load_sar2rgb_sup_records(resolved_sup)
                self._records.extend(sup_records)
                logging.getLogger(__name__).info(
                    f"Added {len(sup_records)} sar2rgb_sup pairs from {resolved_sup}"
                )
            elif sar2rgb_sup_manifest:
                logging.getLogger(__name__).warning(
                    "sar2rgb_sup_manifest not found at %s – skipping",
                    sar2rgb_sup_manifest,
                )

        # Filter out excluded samples (bad_samples + paired val paths when train)
        exclude = _load_exclude_set(exclude_file)
        if split == "train" and paired_val_manifest:
            exclude = exclude | _load_paired_val_exclude_set(paired_val_manifest)
        if exclude:
            before = len(self._records)
            self._records = [
                r for r in self._records
                if str(Path(r["input_path"]).resolve()) not in exclude
                and (not with_target or str(Path(r["target_path"]).resolve()) not in exclude)
            ]
            after = len(self._records)
            if before != after:
                logging.getLogger(__name__).info(
                    f"Excluded {before - after} samples via exclude set "
                    f"({after} remaining)"
                )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(source, target)`` tensors in [0, 1].

        For val/test splits where no target is available the second element is a
        zero tensor with the correct shape.
        """
        rec = self._records[idx]

        use_random_crop = self.use_random_crop
        use_resize_mode = self.load_size is not None and self.load_size != self.resolution
        use_resize_and_crop = use_resize_mode and self.load_size > self.resolution

        if use_random_crop:
            if use_resize_and_crop:
                # Resize-and-crop: crop from load_size x load_size
                max_offset = self.load_size - self.resolution
                crop_x = random.randint(0, max_offset)
                crop_y = random.randint(0, max_offset)
                crop_pos = (crop_x, crop_y)
            elif not use_resize_mode:
                # Direct crop: get image size for crop bounds (assume source/target same size)
                with Image.open(rec["input_path"]) as tmp:
                    w, h = tmp.size
                if w >= self.resolution and h >= self.resolution:
                    crop_x = random.randint(0, w - self.resolution)
                    crop_y = random.randint(0, h - self.resolution)
                    crop_pos = (crop_x, crop_y)
                else:
                    crop_pos = None
                    use_random_crop = False
            else:
                # use_resize_mode and load_size < resolution: resize only, no crop
                crop_pos = None
                use_random_crop = False
        else:
            crop_pos = None

        load_sz = self.load_size if use_resize_mode else None
        source = _load_image_as_tensor(
            rec["input_path"],
            self.source_channels,
            self.resolution,
            crop_pos=crop_pos,
            use_random_crop=use_random_crop,
            load_size=load_sz,
        )
        if self.use_sar_despeckle:
            source = _despeckle_tensor(
                source,
                kernel_size=self.sar_despeckle_kernel_size,
                strength=self.sar_despeckle_strength,
            )

        if self.with_target:
            target = _load_image_as_tensor(
                rec["target_path"],
                self.target_channels,
                self.resolution,
                crop_pos=crop_pos,
                use_random_crop=use_random_crop,
                load_size=load_sz,
            )
        else:
            target = torch.zeros(self.target_channels, self.resolution, self.resolution)

        # Apply random flip augmentations consistently to both source and target
        if self.use_horizontal_flip and torch.rand(1).item() > 0.5:
            source = TF.hflip(source)
            target = TF.hflip(target)
        if self.use_vertical_flip and torch.rand(1).item() > 0.5:
            source = TF.vflip(source)
            target = TF.vflip(target)

        return source, target
