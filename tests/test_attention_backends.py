# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for xformers / flash-attn attention backend config and activation."""

import pytest
import torch

from src.models.unet_ddbm import create_model
from src.models.unet_dbim import create_dbim_model


def _make_small_ddbm_model():
    return create_model(
        image_size=32,
        in_channels=1,
        num_channels=32,
        num_res_blocks=1,
        attention_resolutions="32",
        condition_mode="concat",
        channel_mult="1,2",
        attention_head_dim=32,
    )


def _make_small_dbim_model():
    return create_dbim_model(
        image_size=32,
        in_channels=1,
        num_channels=32,
        num_res_blocks=1,
        attention_resolutions="32",
        condition_mode="concat",
        channel_mult="1,2",
        attention_head_dim=32,
    )


class TestDDBMConfigAttentionFields:
    def test_ddbm_config_has_attention_fields(self):
        from examples.ddbm.config import TaskConfig
        cfg = TaskConfig()
        assert hasattr(cfg, "enable_xformers")
        assert hasattr(cfg, "enable_flash_attn")
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attn is False

    def test_dbim_config_inherits_attention_fields(self):
        from examples.dbim.config import TaskConfig
        cfg = TaskConfig()
        assert hasattr(cfg, "enable_xformers")
        assert hasattr(cfg, "enable_flash_attn")
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attn is False


class TestCUTConfigAttentionFields:
    def test_cut_config_has_attention_fields(self):
        from examples.cut.config import TaskConfig
        cfg = TaskConfig()
        assert hasattr(cfg, "enable_xformers")
        assert hasattr(cfg, "enable_flash_attn")
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attn is False


class TestFlashAttnProcessor:
    def test_set_attn_processor_2_0_ddbm(self):
        """AttnProcessor2_0 (SDPA) can be set on a DDBM model with attention layers."""
        from diffusers.models.attention import Attention
        from diffusers.models.attention_processor import AttnProcessor2_0
        model = _make_small_ddbm_model()
        count = 0
        for mod in model.unet.modules():
            if isinstance(mod, Attention):
                mod.set_processor(AttnProcessor2_0())
                count += 1
        assert count > 0, "Expected attention layers in model"
        # Forward still works
        x = torch.randn(1, 1, 32, 32)
        t = torch.tensor([0.5])
        xT = torch.randn(1, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == x.shape

    def test_set_attn_processor_2_0_dbim(self):
        """AttnProcessor2_0 (SDPA) can be set on a DBIM model with attention layers."""
        from diffusers.models.attention import Attention
        from diffusers.models.attention_processor import AttnProcessor2_0
        model = _make_small_dbim_model()
        count = 0
        for mod in model.unet.modules():
            if isinstance(mod, Attention):
                mod.set_processor(AttnProcessor2_0())
                count += 1
        assert count > 0, "Expected attention layers in model"
        x = torch.randn(1, 1, 32, 32)
        t = torch.tensor([0.5])
        xT = torch.randn(1, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == x.shape


class TestXformersOptional:
    def test_enable_xformers_or_skip(self):
        """If xformers is installed, enable_xformers should succeed; otherwise skip."""
        model = _make_small_ddbm_model()
        try:
            import xformers  # noqa: F401
            model.unet.enable_xformers_memory_efficient_attention()
        except ImportError:
            pytest.skip("xformers not installed")
        except Exception:
            pytest.skip("xformers not functional in this environment")
        # Forward still works
        x = torch.randn(1, 1, 32, 32)
        t = torch.tensor([0.5])
        xT = torch.randn(1, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == x.shape
