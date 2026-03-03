# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Core DBIM trainer for MAVIC-T tasks.

DBIM shares DDBM's training objective and differs mainly in sampling.
This trainer reuses the DDBM training loop with DBIM model/scheduler/pipeline
components.
"""

from __future__ import annotations

from pathlib import Path

from accelerate.logging import get_logger

from src.models.unet_dbim import create_dbim_model
from src.schedulers import DBIMScheduler

from examples.ddbm.trainer import DDBMTrainer
from examples.ddbm.dataset_wrapper import resolve_paired_val_manifest
from .dataset_wrapper import MavicTDBIMDataset, PairedValDataset


logger = get_logger(__name__, log_level="INFO")


class DBIMTrainer(DDBMTrainer):
    """End-to-end DBIM trainer driven by :class:`examples.dbim.config.TaskConfig`."""

    @property
    def baseline_name(self) -> str:
        return "dbim"

    @property
    def pipeline_class_name(self) -> str:
        return "DBIMPipeline"

    def get_validation_pipelines(self):
        from src.pipelines.dbim import DBIMPipeline, DBIMLatentPipeline

        return DBIMPipeline, DBIMLatentPipeline

    def get_inference_kwargs(self, source_inp):
        cfg = self.cfg
        kwargs = {
            "source_image": source_inp,
            "num_inference_steps": cfg.num_inference_steps,
            "sampler": cfg.sampler,
            "guidance": cfg.guidance,
            "churn_step_ratio": cfg.churn_step_ratio,
            "eta": cfg.eta,
            "order": cfg.order,
            "lower_order_final": cfg.lower_order_final,
            "clip_denoised": cfg.clip_denoised,
            "output_type": "pt",
        }
        if not cfg.use_latent_target:
            kwargs["cfg_scale"] = getattr(cfg, "cfg_scale", 1.0)
        return kwargs

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        if self.cfg.use_latent_target:
            src_ch = self.cfg.source_channels
            tgt_ch = self.cfg.target_channels
        else:
            src_ch = self.cfg.model_channels
            tgt_ch = self.cfg.model_channels

        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )

        train_ds = MavicTDBIMDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            source_channels=src_ch,
            target_channels=tgt_ch,
            use_augmented=self.cfg.use_augmented,
            use_random_crop=getattr(self.cfg, "use_random_crop", False),
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
            paired_val_manifest=paired_val_manifest_str,
            sar2rgb_sup_manifest=self.cfg.sar2rgb_sup_manifest if getattr(self.cfg, "use_sar2rgb_sup", False) else None,
        )
        val_ds = None
        if (
            (self.cfg.validation_epochs is not None and self.cfg.validation_epochs > 0)
            or (self.cfg.validation_steps is not None and self.cfg.validation_steps > 0)
        ):
            val_resolution = getattr(self.cfg, "output_resolution", None) or self.cfg.resolution
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=val_resolution,
                    source_channels=src_ch,
                    target_channels=tgt_ch,
                )
                logger.info(
                    "Using paired val set for validation: %s (%d pairs)",
                    resolved_paired,
                    len(val_ds),
                )
            elif getattr(self.cfg, "paired_val_manifest", None):
                logger.warning(
                    "Paired val manifest not found at %s (tried cwd and project root) – falling back to test split",
                    self.cfg.paired_val_manifest,
                )
            if val_ds is None:
                try:
                    val_ds = MavicTDBIMDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=val_resolution,
                        source_channels=src_ch,
                        target_channels=tgt_ch,
                        with_target=False,
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning(
                        "Test split unavailable for %s - skipping validation",
                        self.cfg.task_name,
                    )
        return train_ds, val_ds

    # ----- model / scheduler -------------------------------------------------

    def build_model(self, image_size: int | None = None):
        in_ch = self.cfg.latent_channels if self.cfg.use_latent_target else self.cfg.model_channels
        if image_size is None:
            image_size = self.cfg.resolution
        return create_dbim_model(
            image_size=image_size,
            in_channels=in_ch,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            unet_type=self.cfg.unet_type,
            attention_resolutions=self.cfg.attention_resolutions,
            dropout=self.cfg.dropout,
            condition_mode=self.cfg.condition_mode,
            channel_mult=self.cfg.channel_mult,
        )

    def build_scheduler(self):
        return DBIMScheduler(
            sigma_min=self.cfg.sigma_min,
            sigma_max=self.cfg.sigma_max,
            sigma_data=self.cfg.sigma_data,
            beta_d=self.cfg.beta_d,
            beta_min=self.cfg.beta_min,
            pred_mode=self.cfg.pred_mode,
            num_train_timesteps=self.cfg.num_inference_steps,
            sampler=self.cfg.sampler,
            eta=self.cfg.eta,
            order=self.cfg.order,
            lower_order_final=self.cfg.lower_order_final,
        )
