# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the DBIM baseline integration."""

import torch

from src.models.unet_dbim import DBIMUNet, create_dbim_model
from src.schedulers import DBIMScheduler
from src.pipelines.dbim import DBIMPipeline, DBIMPipelineOutput


class TestDBIMScheduler:
    def test_default_config(self):
        sched = DBIMScheduler()
        assert sched.config.sampler == "dbim"
        assert sched.config.eta == 1.0
        assert sched.config.order == 2

    def test_set_timesteps_dbim(self):
        sched = DBIMScheduler(sigma_min=0.01, sigma_max=1.0)
        sched.set_timesteps(8, sampler="dbim")
        assert sched.sigmas is not None
        assert len(sched.sigmas) == 9  # num_inference_steps + 1

    def test_set_timesteps_heun(self):
        sched = DBIMScheduler(sigma_min=0.01, sigma_max=1.0)
        sched.set_timesteps(8, sampler="heun")
        assert sched.sigmas is not None
        assert sched.sigmas[-1].item() == 0.0

    def test_get_abc_shapes(self):
        sched = DBIMScheduler(sigma_min=0.01, sigma_max=1.0)
        t = torch.tensor([0.9, 0.3], dtype=torch.float32)
        a_t, b_t, c_t = sched.get_abc(t)
        assert a_t.shape == t.shape
        assert b_t.shape == t.shape
        assert c_t.shape == t.shape
        assert torch.isfinite(a_t).all()
        assert torch.isfinite(b_t).all()
        assert torch.isfinite(c_t).all()


class TestDBIMPipeline:
    def _make_pipeline(self) -> DBIMPipeline:
        model = create_dbim_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
            channel_mult="1",
        )
        assert isinstance(model, DBIMUNet)
        scheduler = DBIMScheduler(
            sigma_min=0.01,
            sigma_max=1.0,
            sigma_data=0.5,
            pred_mode="vp",
        )
        return DBIMPipeline(unet=model, scheduler=scheduler)

    def test_dbim_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        result = pipe(
            source_image=source,
            sampler="dbim",
            num_inference_steps=4,
            output_type="pt",
        )
        assert isinstance(result, DBIMPipelineOutput)
        assert result.images.shape == source.shape
        assert result.nfe == 5
        assert result.sampler == "dbim"

    def test_dbim_high_order_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        result = pipe(
            source_image=source,
            sampler="dbim_high_order",
            order=2,
            num_inference_steps=4,
            output_type="pt",
        )
        assert result.images.shape == source.shape
        assert result.nfe > 0
        assert result.sampler == "dbim_high_order"

    def test_heun_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        result = pipe(
            source_image=source,
            sampler="heun",
            churn_step_ratio=0.0,
            num_inference_steps=4,
            output_type="pt",
        )
        assert result.images.shape == source.shape
        assert result.nfe > 0
        assert result.sampler == "heun"
