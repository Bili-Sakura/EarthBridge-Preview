# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for pixel-space DDBM CFG behavior."""

import torch

from src.models.unet_ddbm import DDBMUNet, create_model
from src.schedulers import DDBMScheduler
from src.pipelines.ddbm import DDBMPipeline, DDBMPipelineOutput


class TestDDBMPipelineCFG:
    def _make_pipeline(self) -> DDBMPipeline:
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
            channel_mult="1",
        )
        assert isinstance(model, DDBMUNet)
        scheduler = DDBMScheduler(
            sigma_min=0.01,
            sigma_max=1.0,
            sigma_data=0.5,
            pred_mode="vp",
        )
        return DDBMPipeline(unet=model, scheduler=scheduler)

    def test_cfg_scale_one_matches_legacy_path(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)

        gen_default = torch.Generator().manual_seed(1234)
        out_default = pipe(
            source_image=source,
            num_inference_steps=4,
            guidance=1.0,
            churn_step_ratio=0.0,
            output_type="pt",
            generator=gen_default,
        )
        assert isinstance(out_default, DDBMPipelineOutput)

        gen_cfg1 = torch.Generator().manual_seed(1234)
        out_cfg1 = pipe(
            source_image=source,
            num_inference_steps=4,
            guidance=1.0,
            cfg_scale=1.0,
            churn_step_ratio=0.0,
            output_type="pt",
            generator=gen_cfg1,
        )

        assert torch.allclose(out_default.images, out_cfg1.images)
        assert out_default.nfe == out_cfg1.nfe

    def test_cfg_scale_gt_one_preserves_shape_and_increases_nfe(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)

        gen_no_cfg = torch.Generator().manual_seed(4321)
        out_no_cfg = pipe(
            source_image=source,
            num_inference_steps=4,
            guidance=1.0,
            cfg_scale=1.0,
            churn_step_ratio=0.0,
            output_type="pt",
            generator=gen_no_cfg,
        )

        gen_cfg = torch.Generator().manual_seed(4321)
        out_cfg = pipe(
            source_image=source,
            num_inference_steps=4,
            guidance=1.0,
            cfg_scale=3.0,
            churn_step_ratio=0.0,
            output_type="pt",
            generator=gen_cfg,
        )

        assert out_cfg.images.shape == out_no_cfg.images.shape
        assert out_cfg.nfe == out_no_cfg.nfe * 2
