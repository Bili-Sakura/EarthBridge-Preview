# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Shared training utilities for MAVIC-T baselines.

Provides:
* :func:`lambda_repa_cosine` – cosine decay for REPA lambda (step-based schedule).
* :func:`create_optimizer` – build Prodigy or Adam/AdamW from config.
* :func:`save_checkpoint_diffusers` – save model weights in diffusers-style
  directory layout (``unet/``, ``scheduler/``, ``model_index.json``) using
  safetensors.
* :func:`push_checkpoint_to_hub` – upload a diffusers-style checkpoint
  directory to the Hugging Face Hub.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


def _multiscale_resolutions(height: int, width: int, base_resolution: int) -> list[int]:
    """Return powers-of-two resolutions for multiscale loss."""
    max_res = min(height, width)
    if base_resolution <= 0:
        raise ValueError("base_resolution must be > 0")

    resolutions: list[int] = []
    if base_resolution <= max_res:
        r = int(base_resolution)
        while r <= max_res:
            resolutions.append(r)
            r *= 2

    if max_res not in resolutions:
        resolutions.append(max_res)
    return sorted(set(resolutions))


def multiscale_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weights: Optional[torch.Tensor] = None,
    base_resolution: int = 32,
) -> torch.Tensor:
    """Compute the paper-style multiscale MSE with 1/s scale weighting."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have same shape, got {pred.shape} vs {target.shape}"
        )
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D BCHW tensors, got ndim={pred.ndim}")

    b, _, h, w = pred.shape
    resolutions = _multiscale_resolutions(h, w, int(base_resolution))

    if sample_weights is None:
        sample_weights = torch.ones(b, device=pred.device, dtype=pred.dtype)
    else:
        sample_weights = sample_weights.to(device=pred.device, dtype=pred.dtype).reshape(b)

    weighted_losses = []
    scale_weights = []
    for s in resolutions:
        pred_down = F.adaptive_avg_pool2d(pred, (s, s))
        target_down = F.adaptive_avg_pool2d(target, (s, s))
        per_sample_mse = (pred_down - target_down).pow(2).mean(dim=(1, 2, 3))
        loss_s = (sample_weights * per_sample_mse).mean()
        scale_w = 1.0 / float(s)
        weighted_losses.append(scale_w * loss_s)
        scale_weights.append(scale_w)

    return sum(weighted_losses) / max(sum(scale_weights), 1e-12)


def _to_yaml_serializable(
    obj: Any,
    visited: Optional[Dict[int, Any]] = None,
    stack: Optional[set[int]] = None,
) -> Any:
    if visited is None:
        visited = {}
    if stack is None:
        stack = set()
    obj_id = id(obj)
    if obj_id in stack:
        return str(obj)
    if obj_id in visited:
        return visited[obj_id]
    stack.add(obj_id)
    if isinstance(obj, dict):
        result = {k: _to_yaml_serializable(v, visited, stack) for k, v in obj.items()}
    elif is_dataclass(obj):
        result = _to_yaml_serializable(asdict(obj), visited, stack)
    elif isinstance(obj, list):
        result = [_to_yaml_serializable(v, visited, stack) for v in obj]
    elif isinstance(obj, tuple):
        result = tuple(_to_yaml_serializable(v, visited, stack) for v in obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        result = obj
    else:
        result = str(obj)
    stack.discard(obj_id)
    visited[obj_id] = result
    return result


# ---------------------------------------------------------------------------
# REPA lambda schedule
# ---------------------------------------------------------------------------


def lambda_repa_cosine(step: int, start: float, end: float, decay_steps: int) -> float:
    """Cosine-style decay for REPA lambda: start at step 0, end at decay_steps."""
    if decay_steps <= 0 or step >= decay_steps:
        return end if decay_steps > 0 else start
    progress = step / decay_steps
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def checkpoint_has_accelerator_state(checkpoint_dir: str | Path) -> bool:
    """Return True if the checkpoint contains full Accelerate state (optimizer, scheduler, etc.).

    Used to distinguish step checkpoints (full state) from epoch checkpoints (model-only).
    """
    p = Path(checkpoint_dir)
    if not p.is_dir():
        return False
    # Accelerate saves optimizer.pt or optimizer.bin (safetensors)
    for name in ("optimizer.pt", "optimizer.bin"):
        if (p / name).is_file():
            return True
    return False


def checkpoint_dir_sort_key(name: str) -> tuple[int, int]:
    """Sort key for checkpoint dirs: (0, step) for checkpoint-{step}, (1, epoch) for checkpoint-epoch-{epoch}.

    Use for sorting and for resume: only dirs with key[0] == 0 (step checkpoints) have full
    accelerator state; checkpoint-epoch-* dirs are model-only saves.
    """
    if not name.startswith("checkpoint"):
        return (2, 0)
    parts = name.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return (0, int(parts[1]))
    if len(parts) >= 3 and parts[1] == "epoch" and parts[2].isdigit():
        return (1, int(parts[2]))
    return (2, 0)


# ---------------------------------------------------------------------------
# Accelerate tracker helpers
# ---------------------------------------------------------------------------


def normalize_accelerate_log_with(log_with: Any) -> Any:
    """Normalize ``Accelerator(log_with=...)`` values from CLI-friendly inputs.

    Supports:
    - single strings: ``"tensorboard"``
    - comma-separated strings: ``"tensorboard,swanlab"``
    - iterable values: ``["tensorboard", "swanlab"]``
    """

    if log_with is None:
        return None

    if isinstance(log_with, str):
        value = log_with.strip()
        if not value:
            return None
        if "," not in value:
            return value
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            return None
        return parts if len(parts) > 1 else parts[0]

    if isinstance(log_with, (list, tuple, set)):
        parts = []
        for item in log_with:
            if item is None:
                continue
            if isinstance(item, str):
                parts.extend([token.strip() for token in item.split(",") if token.strip()])
            else:
                token = str(item).strip()
                if token:
                    parts.append(token)
        if not parts:
            return None
        return parts if len(parts) > 1 else parts[0]

    return log_with


def _accelerate_uses_swanlab(log_with: Any) -> bool:
    normalized = normalize_accelerate_log_with(log_with)
    if normalized is None:
        return False
    if isinstance(normalized, str):
        return normalized.lower() in {"swanlab", "all"}
    return any(str(item).strip().lower() in {"swanlab", "all"} for item in normalized)


def _parse_csv_values(raw_value: Any) -> Optional[list[str]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        values = [item.strip() for item in raw_value.split(",") if item.strip()]
        return values or None
    if isinstance(raw_value, (list, tuple, set)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
        return values or None
    value = str(raw_value).strip()
    return [value] if value else None


def build_accelerate_tracker_config(cfg: Any) -> Dict[str, str]:
    """Build a tracker-safe config dict from a config object."""

    if hasattr(cfg, "__dict__"):
        return {k: str(v) for k, v in vars(cfg).items()}
    if is_dataclass(cfg):
        return {k: str(v) for k, v in asdict(cfg).items()}
    raise TypeError(f"Cannot build tracker config from type: {type(cfg).__name__}")


def build_accelerate_tracker_init_kwargs(cfg: Any, project_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Build ``accelerator.init_trackers(..., init_kwargs=...)`` payload.

    For SwanLab, this follows Accelerate's native integration path:
    ``init_kwargs={"swanlab": {...}}``.
    """

    if not _accelerate_uses_swanlab(getattr(cfg, "log_with", None)):
        return None

    swanlab_kwargs: Dict[str, Any] = {
        # Default to project_name for stable and predictable run naming.
        "experiment_name": getattr(cfg, "swanlab_experiment_name", None) or project_name,
    }

    swanlab_description = getattr(cfg, "swanlab_description", None)
    if swanlab_description:
        swanlab_kwargs["description"] = str(swanlab_description)

    swanlab_tags = _parse_csv_values(getattr(cfg, "swanlab_tags", None))
    if swanlab_tags:
        swanlab_kwargs["tags"] = swanlab_tags

    raw_json = getattr(cfg, "swanlab_init_kwargs_json", None)
    if raw_json:
        try:
            parsed_kwargs = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid swanlab_init_kwargs_json: expected a JSON object string."
            ) from exc
        if not isinstance(parsed_kwargs, dict):
            raise ValueError(
                "Invalid swanlab_init_kwargs_json: expected a JSON object."
            )
        # JSON payload takes final precedence for advanced customizations.
        swanlab_kwargs.update(parsed_kwargs)

    swanlab_kwargs = {k: v for k, v in swanlab_kwargs.items() if v is not None}
    if not swanlab_kwargs:
        return None
    return {"swanlab": swanlab_kwargs}


def log_validation_images_to_trackers(
    accelerator: Any,
    images: Any,
    step: int,
    tag: str = "validation/samples",
) -> None:
    """Log validation sample images to all active experiment trackers (TensorBoard, WandB, SwanLab, etc.).

    Images are already saved locally by the trainer; this sends the same grid/samples
    to the configured trackers so they appear in the experiment UI.

    Args:
        accelerator: HuggingFace Accelerator instance (with init_trackers already called).
        images: Single PIL Image or list of PIL Images to log (e.g. a grid or per-batch grids).
        step: Global step to associate with the logged images.
        tag: Key under which to log (e.g. "validation/samples").
    """
    if not accelerator.is_main_process:
        return
    try:
        from PIL import Image as PILImage
    except ImportError:
        return
    if isinstance(images, PILImage.Image):
        images = [images]
    if not images:
        return
    # Ensure list of PIL for trackers that expect it
    out: list = []
    for img in images:
        if isinstance(img, PILImage.Image):
            out.append(img)
        elif isinstance(img, (str, Path)):
            out.append(PILImage.open(img).convert("RGB"))
        else:
            out.append(PILImage.fromarray(np.asarray(img)).convert("RGB") if hasattr(img, "__array__") else img)
    if not out:
        return
    values = {tag: out}
    tracker_names = ("tensorboard", "wandb", "swanlab", "aim", "comet_ml", "mlflow", "clearml", "dvclive", "trackio")
    for name in tracker_names:
        try:
            t = accelerator.get_tracker(name, unwrap=True)
            if t is not None and hasattr(t, "log_images"):
                t.log_images(values, step=step)
        except (AttributeError, ValueError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_muon_optimizers():
    """Load Muon optimizer classes, raising ImportError when unavailable."""
    try:
        import muon as muon_module
    except ImportError as exc:
        raise ImportError(
            "Muon optimizer requires the Muon package (https://github.com/KellerJordan/Muon). "
            "Install it with: pip install git+https://github.com/KellerJordan/Muon.git"
        ) from exc
    if not hasattr(muon_module, "Muon") or not hasattr(muon_module, "SingleDeviceMuon"):
        raise ImportError(
            "The installed 'muon' package does not expose Muon optimizers. "
            "Install the optimizer build with: pip install git+https://github.com/KellerJordan/Muon.git"
        )
    return muon_module.Muon, muon_module.SingleDeviceMuon


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    optimizer_type: str = "prodigy",
    lr: float = 1.0,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    prodigy_d0: float = 1e-6,
) -> torch.optim.Optimizer:
    """Create an optimizer from a string identifier.

    Parameters
    ----------
    params : iterable of ``torch.nn.Parameter``
        Model parameters to optimise.
    optimizer_type : str
        ``"prodigy"`` (default) or ``"adamw"`` / ``"adam"`` / ``"muon"``.
    lr : float
        Learning rate.  For Prodigy the recommended value is ``1.0``.
    weight_decay : float
        Weight-decay coefficient.
    betas : tuple
        Beta coefficients for Adam-family optimizers.
    prodigy_d0 : float
        Prodigy d0 parameter (initial estimate of D). Default is 1e-6.

    Returns
    -------
    torch.optim.Optimizer
    """
    name = optimizer_type.lower()
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError(
                "prodigyopt is required for the Prodigy optimizer. "
                "Install it with: pip install prodigyopt"
            )
        return Prodigy(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            d0=prodigy_d0,
        )
    elif name == "muon":
        Muon, SingleDeviceMuon = _load_muon_optimizers()
        # Muon constructor asserts params is a list and sorts it by size, so materialize any generator.
        params_list = list(params)
        try:
            use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Muon optimizer using single-device fallback: %s", exc)
            use_distributed = False
        # Muon paper recommends momentum around 0.95; fall back to that if betas is None.
        momentum = betas[0] if betas else 0.95
        if use_distributed:
            return Muon(params_list, lr=lr, weight_decay=weight_decay, momentum=momentum)
        return SingleDeviceMuon(params_list, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")


# ---------------------------------------------------------------------------
# Diffusers-style checkpoint saving (safetensors)
# ---------------------------------------------------------------------------


def save_training_config(cfg: Any, save_dir: str, *, filename: str = "config.yaml") -> None:
    """Persist the training configuration to a YAML file inside *save_dir*.

    The file is named ``config.yaml`` by default but can be overridden via the
    *filename* parameter. Circular references are mitigated by tracking visited
    objects during serialization. This helper is intended for single-threaded
    use; concurrent callers writing to the same destination should coordinate
    with locks to avoid file-system race conditions.
    """

    os.makedirs(save_dir, exist_ok=True)
    if is_dataclass(cfg):
        data = asdict(cfg)
    elif isinstance(cfg, dict):
        data = cfg
    else:
        try:
            # Skip private attributes when serializing arbitrary objects.
            data = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        except TypeError as exc:
            # Objects without a __dict__ (e.g. using __slots__) fall back to a
            # string representation to ensure persistence still succeeds.
            logger.warning(
                "Falling back to string representation when saving config of type %s (vars() failed: %s).",
                type(cfg).__name__,
                exc,
            )
            data = {"raw_config": str(cfg)}
    data = _to_yaml_serializable(data)
    config_path = Path(save_dir) / filename
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _save_safetensors(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """Save a state dict in safetensors format."""
    from safetensors.torch import save_file
    # Ensure all tensors are contiguous and on CPU, filter out non-tensor values
    cpu_sd = {
        k: v.contiguous().cpu() 
        for k, v in state_dict.items() 
        if isinstance(v, torch.Tensor)
    }
    if not cpu_sd:
        logger.warning(f"No tensor values found in state_dict for {path}, skipping save.")
        return
    save_file(cpu_sd, path)


def save_checkpoint_diffusers(
    save_dir: str,
    model: torch.nn.Module,
    scheduler: Optional[Any] = None,
    *,
    model_name: str = "unet",
    pipeline_class_name: Optional[str] = None,
    extra_state_dicts: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    model_index: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint in diffusers-style directory layout.

    Creates the following structure under *save_dir*::

        save_dir/
            model_index.json
            <model_name>/
                config.json
                diffusion_pytorch_model.safetensors
            scheduler/
                scheduler_config.json

    When the *model* is a :class:`~diffusers.ModelMixin` instance its
    ``save_pretrained`` method is used, producing a ``config.json`` that
    is compatible with ``from_pretrained``.

    Parameters
    ----------
    save_dir : str
        Root directory for the checkpoint.
    model : torch.nn.Module
        The main model whose ``state_dict()`` is saved.
    scheduler : optional
        A diffusers-compatible scheduler with a ``save_config`` method.
    model_name : str
        Sub-directory name for the main model (default ``"unet"``).
    pipeline_class_name : str, optional
        Name of the pipeline class to write in ``model_index.json``
        (e.g. ``"DDBMPipeline"``).  Defaults to the model class name.
    extra_state_dicts : dict, optional
        Additional ``{folder_name: state_dict}`` to save alongside the main
        model (e.g. ``{"vae": vae.state_dict()}``).
    model_index : dict, optional
        Custom content for ``model_index.json``.  When *None* a minimal
        default is written.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- main model ----
    model_dir = os.path.join(save_dir, model_name)
    if hasattr(model, "save_pretrained"):
        # ModelMixin path – writes config.json + safetensors in one call
        model.save_pretrained(model_dir)
    else:
        os.makedirs(model_dir, exist_ok=True)
        _save_safetensors(model.state_dict(), os.path.join(model_dir, "diffusion_pytorch_model.safetensors"))
        # Write a minimal config.json for the model sub-folder
        model_config: Dict[str, Any] = {}
        if hasattr(model, "config") and hasattr(model.config, "to_dict"):
            model_config = model.config.to_dict()
        elif hasattr(model, "config") and isinstance(model.config, dict):
            model_config = model.config
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2, default=str)

    # ---- scheduler ----
    if scheduler is not None:
        sched_dir = os.path.join(save_dir, "scheduler")
        os.makedirs(sched_dir, exist_ok=True)
        if hasattr(scheduler, "save_config"):
            scheduler.save_config(sched_dir)
        elif hasattr(scheduler, "state_dict"):
            sched_config: Dict[str, Any] = {}
            if hasattr(scheduler, "config") and hasattr(scheduler.config, "to_dict"):
                sched_config = scheduler.config.to_dict()
            with open(os.path.join(sched_dir, "scheduler_config.json"), "w") as f:
                json.dump(sched_config, f, indent=2, default=str)

    # ---- extra state dicts ----
    if extra_state_dicts:
        for folder, sd in extra_state_dicts.items():
            d = os.path.join(save_dir, folder)
            os.makedirs(d, exist_ok=True)
            _save_safetensors(sd, os.path.join(d, "diffusion_pytorch_model.safetensors"))

    # ---- model_index.json ----
    if model_index is None:
        cls_name = pipeline_class_name if pipeline_class_name else type(model).__name__
        model_index = {
            "_class_name": cls_name,
            "_diffusers_version": "0.36.0",
            model_name: [type(model).__module__, type(model).__name__],
        }
        if scheduler is not None:
            model_index["scheduler"] = [type(scheduler).__module__, type(scheduler).__name__]
    with open(os.path.join(save_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2, default=str)

    logger.info(f"Saved diffusers-style checkpoint to {save_dir}")


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------

def push_checkpoint_to_hub(
    save_dir: str,
    hub_model_id: str,
    commit_message: str = "Update checkpoint",
    token: Optional[str] = None,
    path_in_repo: Optional[str] = None,
    request_timeout: float = 300.0,
) -> None:
    """Upload a checkpoint directory or file to the Hugging Face Hub.

    Parameters
    ----------
    save_dir : str
        Local checkpoint path. Can be either a directory or a single file.
    hub_model_id : str
        Repository ID on the Hub (e.g. ``"user/model-name"``).
    commit_message : str
        Git commit message for the upload.
    token : str, optional
        Hugging Face API token.  When *None* the token cached by
        ``huggingface-cli login`` is used.
    path_in_repo : str, optional
        Destination subfolder inside the Hub repository.  Use this to
        organise checkpoints by baseline and task, e.g.
        ``"ddbm/sar2eo/checkpoint-epoch-5"``.  When *None* the files are
        uploaded to the repository root (legacy behaviour).
    request_timeout : float, optional
        Timeout in seconds for Hub HTTP requests (upload/commit). Default 300.
        Increase for slow networks or large checkpoints to avoid ReadTimeout.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub is not installed – skipping push to hub.")
        return

    save_path = Path(save_dir)
    if not save_path.exists():
        logger.warning("Checkpoint path does not exist – skipping push to hub: %s", save_dir)
        return

    # Use a longer timeout for upload/commit to avoid ReadTimeout on slow or busy Hub
    prev_timeout = None
    if request_timeout > 0:
        try:
            import huggingface_hub.constants as hf_constants
            from huggingface_hub.utils import reset_sessions
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(request_timeout))
            prev_timeout = getattr(hf_constants, "HF_HUB_DOWNLOAD_TIMEOUT", None)
            hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = request_timeout
            reset_sessions()
        except ImportError:
            pass

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=hub_model_id, exist_ok=True)
        if save_path.is_dir():
            api.upload_folder(
                repo_id=hub_model_id,
                folder_path=str(save_path),
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
        else:
            target_path = path_in_repo if path_in_repo is not None else save_path.name
            api.upload_file(
                repo_id=hub_model_id,
                path_or_fileobj=str(save_path),
                path_in_repo=target_path,
                commit_message=commit_message,
            )
        logger.info(f"Pushed checkpoint to hub: {hub_model_id} (path_in_repo={path_in_repo})")
    finally:
        if prev_timeout is not None:
            import huggingface_hub.constants as hf_constants
            from huggingface_hub.utils import reset_sessions
            hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = prev_timeout
            reset_sessions()
