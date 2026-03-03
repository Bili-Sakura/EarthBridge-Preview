# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for checkpoint pushes to Hugging Face Hub."""

from __future__ import annotations

import sys
import types

from src.utils.training_utils import push_checkpoint_to_hub


def _install_fake_hf_api(monkeypatch):
    calls = {
        "create_repo": [],
        "upload_folder": [],
        "upload_file": [],
    }

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, repo_id, exist_ok=True):
            calls["create_repo"].append((repo_id, exist_ok))

        def upload_folder(self, repo_id, folder_path, path_in_repo=None, commit_message=None):
            calls["upload_folder"].append((repo_id, folder_path, path_in_repo, commit_message))

        def upload_file(self, repo_id, path_or_fileobj, path_in_repo, commit_message=None):
            calls["upload_file"].append((repo_id, path_or_fileobj, path_in_repo, commit_message))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    return calls


def test_push_checkpoint_directory_uses_upload_folder(tmp_path, monkeypatch):
    calls = _install_fake_hf_api(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "weights.safetensors").write_text("dummy", encoding="utf-8")

    push_checkpoint_to_hub(
        save_dir=str(checkpoint_dir),
        hub_model_id="test-user/test-model",
        commit_message="directory checkpoint",
        path_in_repo="ddbm/sar2eo/checkpoint-10",
    )

    assert calls["create_repo"] == [("test-user/test-model", True)]
    assert calls["upload_folder"] == [
        (
            "test-user/test-model",
            str(checkpoint_dir),
            "ddbm/sar2eo/checkpoint-10",
            "directory checkpoint",
        )
    ]
    assert calls["upload_file"] == []


def test_push_checkpoint_file_uses_upload_file(tmp_path, monkeypatch):
    calls = _install_fake_hf_api(monkeypatch)
    checkpoint_file = tmp_path / "model_100.pkl"
    checkpoint_file.write_text("dummy", encoding="utf-8")

    push_checkpoint_to_hub(
        save_dir=str(checkpoint_file),
        hub_model_id="test-user/test-model",
        commit_message="file checkpoint",
        path_in_repo=None,
    )

    assert calls["create_repo"] == [("test-user/test-model", True)]
    assert calls["upload_folder"] == []
    assert calls["upload_file"] == [
        (
            "test-user/test-model",
            str(checkpoint_file),
            "model_100.pkl",
            "file checkpoint",
        )
    ]


def test_push_checkpoint_missing_path_skips_upload(tmp_path, monkeypatch):
    calls = _install_fake_hf_api(monkeypatch)
    missing_path = tmp_path / "missing-checkpoint"

    push_checkpoint_to_hub(
        save_dir=str(missing_path),
        hub_model_id="test-user/test-model",
        commit_message="missing checkpoint",
        path_in_repo="unused/path",
    )

    assert calls["create_repo"] == []
    assert calls["upload_folder"] == []
    assert calls["upload_file"] == []
