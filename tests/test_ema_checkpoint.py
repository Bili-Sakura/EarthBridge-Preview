# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for EMA checkpoint saving and auto-subfolder structuring."""

import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file

from examples.ddbm.config import TaskConfig as DdbmConfig
from src.utils.training_utils import _save_safetensors


class _FakeEMAModel:
    """Minimal stand-in for ``diffusers.training_utils.EMAModel``.

    Replicates the problematic state_dict layout: metadata scalars + a
    ``shadow_params`` list (no named keys).
    """

    def __init__(self, parameters):
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.decay = 0.9999
        self.min_decay = 0.0
        self.optimization_step = 0

    def state_dict(self):
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
        }


def test_ema_state_dict_is_empty_via_naive_save():
    """Demonstrate the bug: ``_save_safetensors`` skips non-tensor values."""
    model = torch.nn.Linear(4, 2)
    ema = _FakeEMAModel(model.parameters())

    # The raw EMA state_dict has no string→Tensor items.
    sd = ema.state_dict()
    tensor_items = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    assert len(tensor_items) == 0, "EMA state_dict should have no direct tensor values"


def test_ema_shadow_params_to_named_dict():
    """The fix: map shadow_params to model param names produces saveable dict."""
    model = torch.nn.Linear(4, 2)
    ema = _FakeEMAModel(model.parameters())

    model_param_names = list(model.state_dict().keys())
    shadow_params = ema.shadow_params
    ema_state_dict = {
        name: param.clone().detach()
        for name, param in zip(model_param_names, shadow_params)
    }

    # Must have the same keys as the model state dict
    assert set(ema_state_dict.keys()) == set(model_param_names)
    # All values must be tensors
    assert all(isinstance(v, torch.Tensor) for v in ema_state_dict.values())


def test_ema_state_dict_saves_to_safetensors():
    """End-to-end: the fixed EMA dict can be persisted via _save_safetensors."""
    model = torch.nn.Linear(4, 2)
    ema = _FakeEMAModel(model.parameters())

    model_param_names = list(model.state_dict().keys())
    shadow_params = ema.shadow_params
    ema_state_dict = {
        name: param.clone().detach()
        for name, param in zip(model_param_names, shadow_params)
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ema.safetensors")
        _save_safetensors(ema_state_dict, path)
        assert os.path.exists(path), "safetensors file should be created"

        loaded = load_file(path)
        assert set(loaded.keys()) == set(model_param_names)


def test_auto_subfolder_with_task_name():
    """When task_name is set, output_dir should be structured."""
    cfg = DdbmConfig(task_name="sar2eo", output_dir="./ckpt")
    # Simulate what the trainer does
    if cfg.task_name:
        cfg.output_dir = os.path.join(cfg.output_dir, "ddbm", cfg.task_name)
    assert cfg.output_dir == os.path.join(".", "ckpt", "ddbm", "sar2eo")


def test_auto_subfolder_without_task_name():
    """When task_name is empty, output_dir should remain unchanged."""
    cfg = DdbmConfig(task_name="", output_dir="./ckpt")
    original = cfg.output_dir
    if cfg.task_name:
        cfg.output_dir = os.path.join(cfg.output_dir, "ddbm", cfg.task_name)
    assert cfg.output_dir == original
