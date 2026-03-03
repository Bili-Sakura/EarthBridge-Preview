# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

import types

import pytest
import torch

import src.utils.training_utils as training_utils
from src.utils.training_utils import create_optimizer


def test_create_optimizer_muon_single_device():
    model = torch.nn.Linear(2, 2)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=0.01,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    assert optimizer.__class__.__name__ == "SingleDeviceMuon"
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["momentum"] == 0.9


def test_create_optimizer_muon_distributed(monkeypatch):
    dummy_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 8,
    )
    monkeypatch.setattr(training_utils, "dist", dummy_dist, raising=False)

    model = torch.nn.Linear(2, 2)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=0.02,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    assert optimizer.__class__.__name__ == "Muon"
    assert optimizer.defaults["lr"] == 0.02
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["momentum"] == 0.9


def test_create_optimizer_muon_world_size_one(monkeypatch):
    dummy_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 1,
    )
    monkeypatch.setattr(training_utils, "dist", dummy_dist, raising=False)

    model = torch.nn.Linear(1, 1)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=0.015,
        weight_decay=0.02,
        betas=(0.95, 0.999),
    )

    assert optimizer.__class__.__name__ == "SingleDeviceMuon"
    assert optimizer.defaults["lr"] == 0.015
    assert optimizer.defaults["weight_decay"] == 0.02
    assert optimizer.defaults["momentum"] == 0.95


def test_normalize_accelerate_log_with_csv_string():
    normalized = training_utils.normalize_accelerate_log_with("tensorboard,swanlab")
    assert normalized == ["tensorboard", "swanlab"]
    assert training_utils.normalize_accelerate_log_with(" swanlab ") == "swanlab"


def test_build_tracker_config_from_namespace():
    cfg = types.SimpleNamespace(task_name="sar2ir", learning_rate=1e-4)
    tracker_config = training_utils.build_accelerate_tracker_config(cfg)
    assert tracker_config == {"task_name": "sar2ir", "learning_rate": "0.0001"}


def test_build_tracker_init_kwargs_for_swanlab_defaults():
    cfg = types.SimpleNamespace(
        log_with="swanlab",
        swanlab_experiment_name=None,
        swanlab_description="baseline run",
        swanlab_tags="stage2,sar2ir",
        swanlab_init_kwargs_json=None,
    )
    init_kwargs = training_utils.build_accelerate_tracker_init_kwargs(cfg, "ddbm-sar2ir")
    assert init_kwargs == {
        "swanlab": {
            "experiment_name": "ddbm-sar2ir",
            "description": "baseline run",
            "tags": ["stage2", "sar2ir"],
        }
    }


def test_build_tracker_init_kwargs_merges_json_override():
    cfg = types.SimpleNamespace(
        log_with="tensorboard,swanlab",
        swanlab_experiment_name="custom-run-name",
        swanlab_description=None,
        swanlab_tags=None,
        swanlab_init_kwargs_json='{"mode":"cloud","experiment_name":"override-name"}',
    )
    init_kwargs = training_utils.build_accelerate_tracker_init_kwargs(cfg, "cut-sar2ir")
    assert init_kwargs["swanlab"]["experiment_name"] == "override-name"
    assert init_kwargs["swanlab"]["mode"] == "cloud"


def test_build_tracker_init_kwargs_returns_none_without_swanlab():
    cfg = types.SimpleNamespace(log_with="tensorboard")
    assert training_utils.build_accelerate_tracker_init_kwargs(cfg, "i2sb-sar2eo") is None


def test_build_tracker_init_kwargs_rejects_non_object_json():
    cfg = types.SimpleNamespace(
        log_with="swanlab",
        swanlab_init_kwargs_json='["unexpected", "list"]',
    )
    with pytest.raises(ValueError, match="JSON object"):
        training_utils.build_accelerate_tracker_init_kwargs(cfg, "turbo-sar2ir")
