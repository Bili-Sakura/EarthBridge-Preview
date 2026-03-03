# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""PyTorch Dataset wrapper around ``MavicTImageToImageDataset``.

This module bridges the HuggingFace ``datasets.Dataset`` returned by
:class:`src.utils.mavic_t_dataset.MavicTImageToImageDataset` and the PyTorch
:class:`torch.utils.data.Dataset` interface expected by DDBM training.

It handles:
* Reading images lazily from disk (via the path columns).
* Resizing to the model resolution.
* Channel adaptation (expanding/repeating channels to ``model_channels``).
* Normalising pixel values to [0, 1] as float32 tensors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

from typing import Set

# Ensure the project root is importable so we can import from ``src``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.mavic_t_dataset import MavicTImageToImageDataset  # noqa: E402


def _load_image_as_tensor(
    path: str,
    channels: int,
    resolution: int,
    crop_pos: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Load an image from *path* and return ``(C, H, W)`` float32 tensor in [0, 1].

    When ``crop_pos`` is provided, a direct crop of size ``resolution`` is used.
    Otherwise the image is resized to ``(resolution, resolution)``.
    """
    with Image.open(path) as img:
        if crop_pos is not None:
            x, y = crop_pos
            if img.width >= x + resolution and img.height >= y + resolution:
                img = img.crop((x, y, x + resolution, y + resolution))
            else:
                img = img.resize((resolution, resolution), Image.BILINEAR)
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


def _sample_random_crop_pos(path: str, crop_size: int) -> Optional[Tuple[int, int]]:
    """Sample a valid top-left crop position for one image."""
    with Image.open(path) as img:
        w, h = img.size
    if w < crop_size or h < crop_size:
        return None
    x = int(torch.randint(0, w - crop_size + 1, (1,)).item())
    y = int(torch.randint(0, h - crop_size + 1, (1,)).item())
    return x, y


def _sample_random_crop_pos_for_pair(
    input_path: str,
    target_path: str,
    crop_size: int,
) -> Optional[Tuple[int, int]]:
    """Sample one crop position valid for both source and target images."""
    with Image.open(input_path) as src:
        src_w, src_h = src.size
    with Image.open(target_path) as tgt:
        tgt_w, tgt_h = tgt.size
    w = min(src_w, tgt_w)
    h = min(src_h, tgt_h)
    if w < crop_size or h < crop_size:
        return None
    x = int(torch.randint(0, w - crop_size + 1, (1,)).item())
    y = int(torch.randint(0, h - crop_size + 1, (1,)).item())
    return x, y


def _despeckle_tensor(
    tensor: torch.Tensor,
    kernel_size: int,
    strength: float,
) -> torch.Tensor:
    """Apply lightweight SAR despeckling via blended local mean filtering.

    The filter uses average pooling to approximate multi-looking and blends it
    with the original signal to preserve edges:
        out = (1 - strength) * x + strength * mean_filter(x)
    """
    if kernel_size <= 1 or strength <= 0.0:
        return tensor

    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    strength = float(max(0.0, min(1.0, strength)))

    pad = kernel_size // 2
    blurred = F.avg_pool2d(tensor.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad).squeeze(0)
    return (1.0 - strength) * tensor + strength * blurred


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


def _load_paired_val_exclude_set(manifest_path: Optional[str]) -> Set[str]:
    """Load all input and target paths from a paired_val manifest as an exclude set.

    The paired val set is a subset of the train pool; exclude these paths when
    loading the train set so train and golden val do not overlap.
    Paths may be absolute or relative to the dataset root (manifest's parent.parent).
    """
    if not manifest_path:
        return set()
    path = Path(manifest_path)
    if not path.is_file():
        return set()
    dataset_root = path.resolve().parent.parent
    out: Set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            for p in parts:
                p = p.strip()
                if p:
                    pp = Path(p)
                    resolved = str((dataset_root / p).resolve() if not pp.is_absolute() else pp.resolve())
                    out.add(resolved)
    return out


def resolve_paired_val_manifest(raw: str | None) -> Path | None:
    """Resolve paired_val_manifest to an existing file (project root preferred, then cwd).
    Prefer project root for relative paths so validation consistently uses pair_val_set
    regardless of CWD; avoids falling back to official val (test split) without ground truth."""
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p if p.is_file() else None
    # Prefer project root first so pair_val_set is used from any working directory
    candidate = _PROJECT_ROOT / p
    if candidate.is_file():
        return candidate
    if p.is_file():
        return p
    return None


def resolve_sar2rgb_sup_manifest(raw: str | None) -> Path | None:
    """Resolve sar2rgb_sup_manifest to an existing file (project root preferred, then cwd)."""
    return resolve_paired_val_manifest(raw)


def _load_sar2rgb_sup_records(manifest_path: Path) -> list[dict]:
    """Load (input_path, target_path) pairs from paired_sar2rgb_sup.txt manifest.

    Manifest format: one line per pair, input_path\\ttarget_path (tab-separated).
    Paths may be absolute or relative to the dataset root (manifest's parent.parent).
    Returns list of dicts with input_path and target_path keys, compatible with MavicTDDBMDataset.
    """
    records = []
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            inp, tgt = parts[0].strip(), parts[1].strip()
            if inp and tgt:
                inp_resolved = _resolve_manifest_path(inp, manifest_path)
                tgt_resolved = _resolve_manifest_path(tgt, manifest_path)
                records.append({"input_path": inp_resolved, "target_path": tgt_resolved})
    return records


class MavicTDDBMDataset(Dataset):
    """A PyTorch :class:`Dataset` that loads MAVIC-T image pairs for DDBM.

    Each ``__getitem__`` returns ``(target_tensor, source_tensor)`` following the
    DDBM convention where ``x0 = target`` and ``x_T = source``.  Both tensors
    are in ``[0, 1]`` range with shape ``(model_channels, resolution, resolution)``.

    Parameters
    ----------
    task : str
        One of ``sar2eo``, ``rgb2ir``, ``sar2ir``, ``sar2rgb``.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    resolution : int
        Spatial resolution to resize images to.
    model_channels : int
        Number of channels the model operates in.
    with_target : bool or None
        Whether to load target images.  Defaults to ``True`` for train, ``False``
        for val/test.
    use_augmented : bool
        If ``True`` and ``split == "train"``, also include the ``*_crop_aug``
        variant as additional samples.
    use_random_crop : bool
        If ``True`` and ``split == "train"``, apply random direct crop of size
        ``resolution`` at runtime (same crop applied to source and target).
    use_horizontal_flip : bool
        If ``True`` and ``split == "train"``, randomly apply horizontal flip
        (applied consistently to both source and target).
    use_vertical_flip : bool
        If ``True`` and ``split == "train"``, randomly apply vertical flip
        (applied consistently to both source and target).
    refined_root, eval_root : str or Path or None
        Forwarded to :class:`MavicTImageToImageDataset`.
    exclude_file : str or None
        Path to a file listing paths to exclude (e.g. bad_samples.txt).
    paired_val_manifest : str or None
        Path to paired_val_<task>.txt. When loading train, paths in this manifest
        are excluded so the train set does not overlap with the golden val set.
    sar2rgb_sup_manifest : str or None
        Path to paired_sar2rgb_sup.txt. When task is sar2rgb and split is train,
        these additional supervised SAR→RGB pairs (OpenEarthMap-SAR, SpaceNet6,
        FUSAR-Map) are appended to the training set.
    """

    def __init__(
        self,
        task: str,
        split: str = "train",
        resolution: int = 512,
        source_channels: Optional[int] = None,
        target_channels: Optional[int] = None,
        model_channels: int = 3,
        with_target: Optional[bool] = None,
        use_augmented: bool = False,
        use_random_crop: bool = False,
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

        # Optionally add sar2rgb_sup supervised pairs (OpenEarthMap-SAR, SpaceNet6, FUSAR-Map)
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
                    "sar2rgb_sup_manifest not found at %s (tried project root and cwd) – skipping",
                    sar2rgb_sup_manifest,
                )

        # Filter out excluded samples (bad_samples.txt and paired val paths)
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

    def get_output_name(self, idx: int) -> str:
        """Return the output filename for the given index (submission format: stem.png)."""
        rec = self._records[idx]
        stem = Path(rec["input_path"]).stem
        return f"{stem}.png"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target, source)`` tensors in [0, 1].

        For val/test splits where no target is available the first element is a
        zero tensor with the correct shape.
        """
        rec = self._records[idx]
        crop_pos = None
        if self.use_random_crop:
            if self.with_target:
                crop_pos = _sample_random_crop_pos_for_pair(
                    rec["input_path"],
                    rec["target_path"],
                    self.resolution,
                )
            else:
                crop_pos = _sample_random_crop_pos(rec["input_path"], self.resolution)

        source = _load_image_as_tensor(
            rec["input_path"],
            self.source_channels,
            self.resolution,
            crop_pos=crop_pos,
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

        return target, source


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> str:
    """Resolve a path from a manifest. If relative, resolve against dataset root (manifest's parent.parent)."""
    p = Path(raw_path)
    if p.is_absolute():
        return raw_path
    dataset_root = manifest_path.resolve().parent.parent
    return str((dataset_root / raw_path).resolve())


class PairedValDataset(Dataset):
    """Dataset that loads (source, target) pairs from a paired validation manifest.

    Manifest format: one line per pair, ``input_path\\ttarget_path`` (tab-separated).
    Paths may be absolute or relative to the dataset root (manifest's parent.parent).
    By default returns ``(target, source)`` tensors in [0, 1] to match MavicTDDBMDataset.
    Use ``return_order="source_target"`` for trainers that expect (source, target) (e.g. CUT).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int,
        source_channels: int,
        target_channels: int,
        return_order: Literal["target_source", "source_target"] = "target_source",
        use_sar_despeckle: bool = False,
        sar_despeckle_kernel_size: int = 5,
        sar_despeckle_strength: float = 0.6,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.return_order = return_order
        self.use_sar_despeckle = use_sar_despeckle
        self.sar_despeckle_kernel_size = max(1, int(sar_despeckle_kernel_size))
        self.sar_despeckle_strength = float(max(0.0, min(1.0, sar_despeckle_strength)))
        self._pairs: list[tuple[str, str]] = []
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Paired val manifest not found: {path}")
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                inp, tgt = parts[0].strip(), parts[1].strip()
                if inp and tgt:
                    inp = _resolve_manifest_path(inp, path)
                    tgt = _resolve_manifest_path(tgt, path)
                    self._pairs.append((inp, tgt))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp_path, tgt_path = self._pairs[idx]
        source = _load_image_as_tensor(inp_path, self.source_channels, self.resolution)
        if self.use_sar_despeckle:
            source = _despeckle_tensor(
                source,
                kernel_size=self.sar_despeckle_kernel_size,
                strength=self.sar_despeckle_strength,
            )
        target = _load_image_as_tensor(tgt_path, self.target_channels, self.resolution)
        if self.return_order == "source_target":
            return source, target
        return target, source
