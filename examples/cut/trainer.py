# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Core CUT trainer for MAVIC-T tasks.

Reference: Park, Taesung, Alexei A. Efros, Richard Zhang, and Jun-Yan Zhu.
“Contrastive Learning for Unpaired Image-to-Image Translation.” ECCV 2020.
https://doi.org/10.1007/978-3-030-58545-7_19.

This module implements the CUT (Contrastive Unpaired Translation) training
loop as a reusable :class:`CUTTrainer` class, following the same structure
as :class:`examples.ddbm.trainer.DDBMTrainer`.  Per-task scripts
instantiate the trainer with their own
:class:`~examples.cut.config.TaskConfig` and can sub-class any method for
task-specific modifications.

The training loop uses *Accelerate* for mixed-precision, multi-GPU, and
gradient-accumulation support, matching the diffusers-style conventions
established in the DDBM baseline.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from diffusers.utils import make_image_grid

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from .config import TaskConfig
from .dataset_wrapper import MavicTCUTDataset

# Resolve paths relative to project root so scripts work regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    """Resolve path; if relative, try project root so CWD-independent."""
    p = Path(path)
    if p.is_absolute() and p.is_file():
        return p
    if p.is_file():
        return p.resolve()
    root_path = _PROJECT_ROOT / p
    if root_path.is_file():
        return root_path
    return p  # return as-is for clearer error messages
from src.models.cut_model import (
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
    GANLoss,
    PatchNCELoss,
)

from src.utils.metrics import MavicCriterion, MetricCalculator  # noqa: E402
from src.utils.training_utils import (  # noqa: E402
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    checkpoint_has_accelerator_state,
    create_optimizer,
    lambda_repa_cosine,
    normalize_accelerate_log_with,
    save_checkpoint_diffusers,
    save_training_config,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


class CUTTrainer:
    """End-to-end CUT trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from .config import sar2eo_config
        from .trainer import CUTTrainer

        cfg = sar2eo_config()
        trainer = CUTTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        """Return ``(train_dataset, val_dataset)``.
        
        The validation loader uses the paired val set when the manifest exists
        (resolved from cwd or project root), otherwise the test split."""
        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )
        train_ds = MavicTCUTDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            load_size=self.cfg.load_size,
            source_channels=self.cfg.source_channels,
            target_channels=self.cfg.target_channels,
            model_channels=self.cfg.model_channels,
            use_augmented=self.cfg.use_augmented,
            use_random_crop=getattr(self.cfg, "use_random_crop", True),
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
            paired_val_manifest=paired_val_manifest_str,
            sar2rgb_sup_manifest=self.cfg.sar2rgb_sup_manifest if getattr(self.cfg, "use_sar2rgb_sup", False) else None,
            use_sar_despeckle=getattr(self.cfg, "use_sar_despeckle", False),
            sar_despeckle_kernel_size=getattr(self.cfg, "sar_despeckle_kernel_size", 5),
            sar_despeckle_strength=getattr(self.cfg, "sar_despeckle_strength", 0.6),
        )
        val_ds = None
        if self.cfg.validation_epochs is not None or self.cfg.validation_steps is not None:
            val_res = self.cfg.validation_resolution if self.cfg.validation_resolution is not None else self.cfg.resolution
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=val_res,
                    source_channels=self.cfg.source_channels,
                    target_channels=self.cfg.target_channels,
                    return_order="source_target",
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
                    val_ds = MavicTCUTDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=val_res,
                        load_size=self.cfg.load_size,
                        source_channels=self.cfg.source_channels,
                        target_channels=self.cfg.target_channels,
                        model_channels=self.cfg.model_channels,
                        with_target=False,
                        use_sar_despeckle=getattr(self.cfg, "use_sar_despeckle", False),
                        sar_despeckle_kernel_size=getattr(self.cfg, "sar_despeckle_kernel_size", 5),
                        sar_despeckle_strength=getattr(self.cfg, "sar_despeckle_strength", 0.6),
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning("Test split unavailable for %s – skipping validation", self.cfg.task_name)
        return train_ds, val_ds

    # ----- model / losses ----------------------------------------------------

    def build_generator(self):
        """Create the CUT generator."""
        if self.cfg.use_latent_target:
            in_ch = self.cfg.latent_channels
            out_ch = in_ch
        else:
            in_ch = self.cfg.source_channels
            out_ch = self.cfg.target_channels
        return create_generator(
            input_nc=in_ch,
            output_nc=out_ch,
            ngf=self.cfg.ngf,
            netG=self.cfg.netG,
            norm_type=self.cfg.normG,
            use_dropout=not self.cfg.no_dropout,
            no_antialias=self.cfg.no_antialias,
            no_antialias_up=self.cfg.no_antialias_up,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_discriminator(self):
        """Create the CUT PatchGAN discriminator."""
        if self.cfg.use_latent_target:
            in_ch = self.cfg.latent_channels
        else:
            in_ch = self.cfg.target_channels
        return create_discriminator(
            input_nc=in_ch,
            ndf=self.cfg.ndf,
            netD=self.cfg.netD,
            n_layers_D=self.cfg.n_layers_D,
            norm_type=self.cfg.normD,
            no_antialias=self.cfg.no_antialias,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_patch_sample_mlp(self):
        """Create the PatchSampleMLP feature network."""
        return create_patch_sample_mlp(
            use_mlp=(self.cfg.netF == "mlp_sample"),
            nc=self.cfg.netF_nc,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    # ----- loss helpers ------------------------------------------------------

    @staticmethod
    def preprocess_batch(batch, device):
        """Scale a ``(source, target)`` batch from [0,1] to [-1,1]."""
        source = batch[0].to(device) * 2 - 1
        target = batch[1].to(device) * 2 - 1
        return source, target

    @staticmethod
    def compute_D_loss(netD, criterion_GAN, real_B, fake_B):
        """Compute discriminator loss."""
        pred_fake = netD(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, False).mean()
        pred_real = netD(real_B)
        loss_D_real = criterion_GAN(pred_real, True).mean()
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    @staticmethod
    def compute_G_loss(
        netG, netD, netF, criterion_GAN, nce_criteria,
        real_A, fake_B, real_B,
        nce_layers, lambda_GAN, lambda_NCE,
        nce_idt, num_patches,
        mavic_criterion=None, mavic_loss_weight=0.1,
        latent_target_encoder=None, lambda_latent=1.0,
        rep_alignment_module=None, lambda_rep_alignment=0.1,
        pixel_target=None, pixel_source=None,
        latent_decode_fn=None, in_latent_space: bool = False,
        source_channels: int = 3, target_channels: int = 3,
    ):
        """Compute generator loss (GAN + NCE + optional identity NCE + optional MAVIC + optional latent + optional REPA).

        Returns
        -------
        loss_G : Tensor
            Total generator loss.
        loss_G_GAN : Tensor or float
            GAN component.
        loss_NCE : Tensor or float
            NCE component.
        loss_NCE_Y : Tensor or float
            Identity NCE component (0.0 if disabled).
        """
        # GAN loss
        if lambda_GAN > 0.0:
            pred_fake = netD(fake_B)
            loss_G_GAN = criterion_GAN(pred_fake, True).mean() * lambda_GAN
        else:
            loss_G_GAN = torch.tensor(0.0, device=real_A.device)

        # NCE loss
        if lambda_NCE > 0.0:
            loss_NCE = CUTTrainer._calculate_NCE_loss(
                netG, netF, nce_criteria, real_A, fake_B, nce_layers, lambda_NCE, num_patches,
                source_channels=source_channels, target_channels=target_channels,
            )
        else:
            loss_NCE = torch.tensor(0.0, device=real_A.device)

        # Identity NCE loss
        loss_NCE_Y = torch.tensor(0.0, device=real_A.device)
        if nce_idt and lambda_NCE > 0.0:
            # Pass real_B through generator; adapt real_B to source_channels when different
            real_B_for_G = real_B
            if not in_latent_space and real_B.shape[1] != source_channels:
                if target_channels < source_channels:
                    real_B_for_G = real_B.repeat(1, source_channels // target_channels, 1, 1)
                else:
                    real_B_for_G = real_B.mean(dim=1, keepdim=True)
            idt_B = netG(real_B_for_G)
            loss_NCE_Y = CUTTrainer._calculate_NCE_loss(
                netG, netF, nce_criteria, real_B_for_G, idt_B, nce_layers, lambda_NCE, num_patches,
                source_channels=source_channels, target_channels=target_channels,
            )
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_G = loss_G_GAN + loss_NCE_both
        extras = {
            "loss_mavic": None,
            "loss_latent": None,
            "loss_rep_alignment": None,
        }

        decoded = None
        if latent_decode_fn is not None and (mavic_criterion is not None or rep_alignment_module is not None):
            decoded = latent_decode_fn(fake_B)
            if pixel_target is not None and decoded.shape[1] != pixel_target.shape[1]:
                if decoded.shape[1] == 3 and pixel_target.shape[1] == 1:
                    decoded = decoded.mean(dim=1, keepdim=True)
                elif decoded.shape[1] == 1 and pixel_target.shape[1] == 3:
                    decoded = decoded.repeat(1, 3, 1, 1)

        # Optional MAVIC metric-based loss
        if mavic_criterion is not None:
            pred_for_metric = decoded if decoded is not None else fake_B
            target_for_metric = pixel_target if pixel_target is not None else real_B
            pred_01 = (pred_for_metric + 1) * 0.5
            target_01 = (target_for_metric + 1) * 0.5
            pred_01 = pred_01.clamp(0, 1)
            target_01 = target_01.clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, target_01)
            loss_G = loss_G + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        # Optional latent-space L2 loss
        if latent_target_encoder is not None and not in_latent_space:
            latent_pred = latent_target_encoder.encode_with_grad(fake_B)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(real_B).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss_G = loss_G + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

        # Optional representation alignment loss (REPA)
        # REPA teacher encodes the *target* (ground-truth) image, not source.
        if rep_alignment_module is not None:
            target_for_enc = pixel_target if pixel_target is not None else real_B
            with torch.no_grad():
                enc_feats = rep_alignment_module.extract_features(target_for_enc)
            rep_features = decoded if decoded is not None else fake_B
            rep_loss = rep_alignment_module.compute_alignment_loss(rep_features, enc_feats)
            # Add lambda_rep_alignment * (rep_loss + 1): offset keeps total loss positive for
            # visualization (rep_loss is negative cosine similarity in [-1, 1]); gradient unchanged.
            loss_G = loss_G + lambda_rep_alignment * (rep_loss + 1.0)
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss_G, loss_G_GAN, loss_NCE, loss_NCE_Y, extras

    @staticmethod
    def _adapt_for_encoder(x: torch.Tensor, actual_ch: int, encoder_ch: int) -> torch.Tensor:
        """Adapt tensor to encoder's expected channel count (for encode_only path)."""
        if actual_ch == encoder_ch:
            return x
        if actual_ch < encoder_ch:
            return x.repeat(1, encoder_ch // actual_ch, 1, 1)
        return x.mean(dim=1, keepdim=True)

    @staticmethod
    def _calculate_NCE_loss(netG, netF, nce_criteria, src, tgt, nce_layers, lambda_NCE, num_patches,
                            source_channels: int = 3, target_channels: int = 3):
        """Compute contrastive loss across multiple encoder layers.
        Encoder expects source_channels; adapt src/tgt when channel counts differ."""
        if src.shape[1] != source_channels:
            src = CUTTrainer._adapt_for_encoder(src, src.shape[1], source_channels)
        if tgt.shape[1] != source_channels:
            tgt = CUTTrainer._adapt_for_encoder(tgt, tgt.shape[1], source_channels)
        n_layers = len(nce_layers)
        feat_q = netG(tgt, nce_layers, encode_only=True)
        feat_k = netG(src, nce_layers, encode_only=True)
        feat_k_pool, sample_ids = netF(feat_k, num_patches, None)
        feat_q_pool, _ = netF(feat_q, num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, nce_criteria):
            loss = crit(f_q, f_k) * lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # ----- linear lr decay scheduler ----------------------------------------

    @staticmethod
    def _load_cut_models_from_checkpoint(netG, netD, netF, load_path: str | Path, accelerator) -> None:
        """Load generator, discriminator, and feature network from a diffusers-style checkpoint."""
        from safetensors.torch import load_file
        load_path = Path(load_path)
        gen_path = load_path / "generator" / "diffusion_pytorch_model.safetensors"
        disc_path = load_path / "discriminator" / "diffusion_pytorch_model.safetensors"
        feat_path = load_path / "feature_network" / "diffusion_pytorch_model.safetensors"
        if not gen_path.is_file():
            raise FileNotFoundError(f"CUT checkpoint missing generator: {gen_path}")
        unwrapped_G = accelerator.unwrap_model(netG)
        unwrapped_D = accelerator.unwrap_model(netD)
        unwrapped_G.load_state_dict(load_file(str(gen_path)), strict=True)
        unwrapped_D.load_state_dict(load_file(str(disc_path)), strict=True)
        if feat_path.is_file():
            netF.load_state_dict(load_file(str(feat_path)), strict=True)

    @staticmethod
    def _get_scheduler(optimizer, n_epochs, n_epochs_decay, last_epoch=-1):
        """Linear LR decay scheduler: constant for n_epochs, then linear to 0."""

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, netG, val_dataloader, accelerator, global_step, latent_target_encoder=None):
        """Generate and save test samples. Optionally evaluate on paired val set with LPIPS/L1/FID."""
        from src.pipelines.cut import CUTPipeline, CUTLatentPipeline

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        was_training = netG.training
        unwrapped = accelerator.unwrap_model(netG)
        unwrapped.eval()

        if latent_target_encoder is not None:
            pipeline = CUTLatentPipeline(generator=unwrapped, vae=latent_target_encoder.vae)
        else:
            pipeline = CUTPipeline(generator=unwrapped)
        pipeline = pipeline.to(accelerator.device)

        sample_dir = Path(cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        first_grid = None

        has_paired_target = isinstance(val_dataloader.dataset, PairedValDataset)
        cols = 3 if has_paired_target else 2

        for batch_idx, batch in enumerate(val_dataloader):
            source, target = batch  # PairedValDataset: (source, target); test split: target is zeros
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                pipeline_kwargs = {
                    "source_image": source_inp,
                    "output_type": "pt",
                }
                if latent_target_encoder is not None:
                    pipeline_kwargs["target_channels"] = cfg.target_channels
                result = pipeline(**pipeline_kwargs)
            generated = (result.images + 1) * 0.5

            src_vis = source_01
            gen_vis = generated
            tgt_vis = target.to(accelerator.device) if has_paired_target else None
            # Collapse 3ch → 1ch when target is 1 channel (e.g. rgb2ir, sar2ir)
            if cfg.target_channels == 1 and gen_vis.shape[1] == 3:
                gen_vis = gen_vis.mean(dim=1, keepdim=True)
            if gen_vis.shape[1] != src_vis.shape[1]:
                if gen_vis.shape[1] == 3 and src_vis.shape[1] == 1:
                    src_vis = src_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and src_vis.shape[1] == 3:
                    gen_vis = gen_vis.repeat(1, 3, 1, 1)
            if tgt_vis is not None and tgt_vis.shape[1] != gen_vis.shape[1]:
                if gen_vis.shape[1] == 3 and tgt_vis.shape[1] == 1:
                    tgt_vis = tgt_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and tgt_vis.shape[1] == 3:
                    gen_vis = gen_vis.repeat(1, 3, 1, 1)

            src_uint8 = (src_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            gen_uint8 = (gen_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            tgt_uint8 = (tgt_vis.clamp(0, 1) * 255).round().to(torch.uint8) if tgt_vis is not None else None

            src_uint8 = src_uint8.permute(0, 2, 3, 1).cpu().numpy()
            gen_uint8 = gen_uint8.permute(0, 2, 3, 1).cpu().numpy()
            tgt_uint8 = tgt_uint8.permute(0, 2, 3, 1).cpu().numpy() if tgt_uint8 is not None else None

            batch_images = []
            batch_size = len(src_uint8)
            for i in range(batch_size):
                src_arr = src_uint8[i]
                gen_arr = gen_uint8[i]
                if src_arr.shape[2] == 1:
                    src_arr = src_arr.squeeze(2)
                if gen_arr.shape[2] == 1:
                    gen_arr = gen_arr.squeeze(2)
                batch_images.append(Image.fromarray(src_arr).convert("RGB"))
                batch_images.append(Image.fromarray(gen_arr).convert("RGB"))
                if tgt_uint8 is not None:
                    tgt_arr = tgt_uint8[i]
                    if tgt_arr.shape[2] == 1:
                        tgt_arr = tgt_arr.squeeze(2)
                    batch_images.append(Image.fromarray(tgt_arr).convert("RGB"))

            grid = make_image_grid(batch_images, rows=batch_size, cols=cols)
            grid.save(sample_dir / f"batch_{batch_idx:03d}.png")
            if first_grid is None:
                first_grid = grid.copy()
            saved += batch_size

        logger.info("Saved %d test sample pairs to %s", saved, sample_dir)

        if first_grid is not None:
            from src.utils.training_utils import log_validation_images_to_trackers
            log_validation_images_to_trackers(accelerator, first_grid, global_step)

        # Evaluate on paired validation set using MAVIC-T metrics (LPIPS, L1, FID)
        metrics_result = {}
        manifest_path = getattr(self, "_resolved_paired_val_manifest", None) or (
            Path(cfg.paired_val_manifest) if getattr(cfg, "paired_val_manifest", None) else None
        )
        if manifest_path is not None and Path(manifest_path).is_file() and accelerator.is_main_process:
            metrics_result = self._evaluate_paired_val_metrics(
                netG, pipeline, accelerator, latent_target_encoder, manifest_path=manifest_path
            )
            if metrics_result:
                logger.info(
                    "Paired val metrics: LPIPS=%.4f L1=%.4f score=%.4f (FID=%s)",
                    metrics_result.get("val_lpips", 0),
                    metrics_result.get("val_l1", 0),
                    metrics_result.get("val_score", 0),
                    metrics_result.get("val_fid", "N/A"),
                )

        if was_training:
            unwrapped.train()
        out = {"saved_samples": saved, "sample_dir": str(sample_dir)}
        out.update(metrics_result)
        return out

    def _evaluate_paired_val_metrics(
        self, netG, pipeline, accelerator, latent_target_encoder, manifest_path=None
    ):
        """Run inference on paired val set and compute MAVIC-T metrics (LPIPS, L1, FID)."""
        cfg = self.cfg
        manifest_path = Path(manifest_path) if manifest_path is not None else Path(cfg.paired_val_manifest)
        if not manifest_path.is_file():
            logger.warning("Paired val manifest not found: %s - skipping metric evaluation", manifest_path)
            return {}

        res = getattr(cfg, "validation_resolution", None) or cfg.resolution
        paired_ds = PairedValDataset(
            manifest_path=manifest_path,
            resolution=res,
            source_channels=cfg.source_channels,
            target_channels=cfg.target_channels,
            return_order="source_target",
        )
        paired_loader = DataLoader(
            paired_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        metric_calc = MetricCalculator(device=str(accelerator.device), compute_fid=False)

        for source, target in paired_loader:
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                pipeline_kwargs = {
                    "source_image": source_inp,
                    "output_type": "pt",
                }
                if latent_target_encoder is not None:
                    pipeline_kwargs["target_channels"] = cfg.target_channels
                result = pipeline(**pipeline_kwargs)
            generated = (result.images + 1) * 0.5

            pred_01 = generated.clamp(0, 1)
            tgt_01 = target.to(accelerator.device).clamp(0, 1)
            if cfg.target_channels == 1 and pred_01.shape[1] == 3:
                pred_01 = pred_01.mean(dim=1, keepdim=True)
            elif pred_01.shape[1] != tgt_01.shape[1]:
                if pred_01.shape[1] == 3 and tgt_01.shape[1] == 1:
                    tgt_01 = tgt_01.repeat(1, 3, 1, 1)
                elif pred_01.shape[1] == 1 and tgt_01.shape[1] == 3:
                    pred_01 = pred_01.repeat(1, 3, 1, 1)
            metric_calc.update(pred_01, tgt_01)

        m = metric_calc.compute()
        return {
            "val_lpips": m.lpips,
            "val_l1": m.l1,
            "val_score": m.score if m.score is not None else m.lpips + m.l1,
        }

    # ----- main training loop ------------------------------------------------

    def train(self):
        """Run the full CUT training loop."""
        cfg = self.cfg

        # Auto-structure checkpoint directory with method/task subfolders (same as DDBM).
        if cfg.task_name:
            cfg.output_dir = os.path.join(cfg.output_dir, "cut", cfg.task_name)

        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if save_model_epochs is not None and save_model_epochs <= 0:
            save_model_epochs = None
        if checkpointing_steps is not None and save_model_epochs is not None:
            logging.warning(
                "checkpointing_steps is set while save_model_epochs is enabled; "
                "epoch checkpoints take priority and step checkpoints will be skipped. "
                "Set save_model_epochs=None to enable step-based checkpointing."
            )
            checkpointing_steps = None

        # Accelerator setup
        logging_dir = os.path.join(cfg.output_dir, "logs")
        # log_with: "tensorboard" | "swanlab" | "wandb" | "all" | "tensorboard,swanlab" etc.
        log_with = normalize_accelerate_log_with(cfg.log_with)
        project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
        # TODO: Multi-GPU validation deadlock – accelerator.wait_for_everyone() / barrier hangs on some
        # setups (e.g. RTX 4090) with "No device id is provided via init_process_group or barrier".
        # InitProcessGroupKwargs does not support device_id; fix distributed sync for validation.
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200), backend="nccl")]
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=log_with,
            project_config=project_config,
            kwargs_handlers=kwargs_handlers,
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        if accelerator.is_main_process:
            os.makedirs(cfg.output_dir, exist_ok=True)

        # Build components
        mavic_criterion = None
        if cfg.use_mavic_loss:
            mavic_criterion = MavicCriterion(
                lpips_weight=cfg.mavic_lpips_weight,
                l1_weight=cfg.mavic_l1_weight,
            )
            logger.info(f"[{cfg.task_name}] Using MAVIC metric loss "
                        f"(lpips_w={cfg.mavic_lpips_weight}, l1_w={cfg.mavic_l1_weight}, "
                        f"loss_w={cfg.mavic_loss_weight})")

        # Latent target encoder (ablation / latent-space training)
        latent_target_encoder = None
        latent_image_size = None
        if cfg.use_latent_target:
            if not cfg.latent_vae_path:
                raise ValueError("use_latent_target=True requires latent_vae_path to be set.")
            from src.utils.latent_target import LatentTargetEncoder
            latent_target_encoder = LatentTargetEncoder(cfg.latent_vae_path)
            vae_scale_factor = 2 ** (len(latent_target_encoder.vae.config.block_out_channels) - 1)
            if cfg.resolution % vae_scale_factor != 0:
                raise ValueError(
                    f"Resolution {cfg.resolution} is not divisible by VAE scale factor "
                    f"{vae_scale_factor} for latent training."
                )
            latent_image_size = cfg.resolution // vae_scale_factor
            logger.info(
                f"[{cfg.task_name}] Using latent target encoder from {cfg.latent_vae_path} "
                f"(lambda={cfg.lambda_latent})"
            )
            logger.info(
                f"[{cfg.task_name}] Latent resolution set to {latent_image_size} "
                f"(vae_scale_factor={vae_scale_factor})"
            )

        if cfg.use_latent_target:
            ch_info = f"latent={cfg.latent_channels}"
        else:
            ch_info = f"in={cfg.source_channels}, out={cfg.target_channels}"
        model_image_size = latent_image_size if latent_image_size is not None else cfg.resolution
        logger.info(f"[{cfg.task_name}] Creating CUT models ({ch_info}, res={model_image_size})")
        netG = self.build_generator()
        netD = self.build_discriminator()
        netF = self.build_patch_sample_mlp()

        # Attention backend: xformers or flash-attn (PyTorch 2.0 SDPA)
        # CUT uses ResNet generators without attention layers, so these are
        # typically no-ops but are accepted for CLI consistency.
        if getattr(cfg, "enable_xformers", False):
            try:
                netG.enable_xformers_memory_efficient_attention()
                logger.info(f"[{cfg.task_name}] Enabled xformers memory-efficient attention on generator")
            except Exception as e:
                logger.warning(
                    "Could not enable xformers on CUT generator (no attention layers?): %s", e,
                )
        elif getattr(cfg, "enable_flash_attn", False):
            try:
                from diffusers.models.attention import Attention
                from diffusers.models.attention_processor import AttnProcessor2_0
                count = 0
                for mod in netG.modules():
                    if isinstance(mod, Attention):
                        mod.set_processor(AttnProcessor2_0())
                        count += 1
                logger.info(f"[{cfg.task_name}] Enabled PyTorch 2.0 SDPA on generator ({count} attention layers)")
            except Exception as e:
                logger.warning(
                    "Could not enable PyTorch 2.0 SDPA on CUT generator: %s", e,
                )

        nce_layers = [int(i) for i in cfg.nce_layers.split(",")]

        criterion_GAN = GANLoss(cfg.gan_mode)
        nce_criteria = [
            PatchNCELoss(nce_T=cfg.nce_T, batch_size=cfg.train_batch_size,
                         nce_includes_all_negatives_from_minibatch=cfg.nce_includes_all_negatives_from_minibatch)
            for _ in nce_layers
        ]

        # Representation alignment (REPA)
        # NOTE: REPA encodes the *target* (ground-truth) image, not the source.
        # Only SAR2RGB is currently supported (MaRS-Base-RGB encodes the RGB target).
        rep_alignment_module = None
        if cfg.use_rep_alignment and cfg.rep_alignment_model_path:
            from src.utils.rep_alignment import MaRSRGBAlignment
            if cfg.task_name == "sar2rgb":
                rep_alignment_module = MaRSRGBAlignment(cfg.rep_alignment_model_path)
            else:
                logger.warning(
                    "REPA is not applicable for task '%s' – no pre-trained "
                    "encoder for the target domain; skipping.", cfg.task_name,
                )
            if rep_alignment_module is not None:
                # Build projector with target_channels (features aligned to target encoder)
                rep_alignment_module.build_projector(cfg.target_channels)
                logger.info(
                    f"[{cfg.task_name}] Representation alignment enabled "
                    f"(model={cfg.rep_alignment_model_path}, "
                    f"lambda={cfg.lambda_rep_alignment}"
                    + (f"→{cfg.lambda_rep_alignment_end} cos decay over {cfg.lambda_rep_alignment_decay_steps} steps" if cfg.lambda_rep_alignment_decay_steps > 0 else "")
                    + ")"
                )

        # Optimisers (G and D share the same lr/beta but are separate)
        g_params = list(netG.parameters())
        if rep_alignment_module is not None and rep_alignment_module.projector is not None:
            g_params += list(rep_alignment_module.projector.parameters())
        optimizer_G = create_optimizer(
            g_params,
            optimizer_type=cfg.optimizer_type,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            prodigy_d0=getattr(cfg, "prodigy_d0", 1e-6),
        )
        optimizer_D = create_optimizer(
            netD.parameters(),
            optimizer_type=cfg.optimizer_type,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            prodigy_d0=getattr(cfg, "prodigy_d0", 1e-6),
        )

        logger.info(f"[{cfg.task_name}] Loading dataset …")
        train_dataset, val_dataset = self.build_datasets()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
        ) if val_dataset is not None else None
        # Disable log_validation on multi-GPU to avoid NCCL barrier deadlock
        if val_dataloader is not None and accelerator.num_processes > 1:
            logger.warning(
                "Disabling log_validation on multi-GPU to avoid NCCL barrier deadlock. "
                "Run validation separately on a single GPU."
            )
            val_dataloader = None

        # Total epochs
        total_epochs = cfg.n_epochs + cfg.n_epochs_decay
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = total_epochs * num_update_steps_per_epoch
        elif total_epochs == 0:
            # Step-based training (n_epochs=0, max_train_steps set): derive epochs from steps
            total_epochs = max(1, math.ceil(cfg.max_train_steps / num_update_steps_per_epoch))

        # LR schedulers
        if cfg.lr_policy == "linear":
            scheduler_G = self._get_scheduler(optimizer_G, cfg.n_epochs, cfg.n_epochs_decay)
            scheduler_D = self._get_scheduler(optimizer_D, cfg.n_epochs, cfg.n_epochs_decay)
        else:
            from diffusers.optimization import get_scheduler as get_lr_scheduler
            scheduler_G = get_lr_scheduler(
                "constant", optimizer=optimizer_G,
                num_warmup_steps=0,
                num_training_steps=cfg.max_train_steps,
            )
            scheduler_D = get_lr_scheduler(
                "constant", optimizer=optimizer_D,
                num_warmup_steps=0,
                num_training_steps=cfg.max_train_steps,
            )

        # Prepare with accelerator
        netG, netD, optimizer_G, optimizer_D, train_dataloader = accelerator.prepare(
            netG, netD, optimizer_G, optimizer_D, train_dataloader,
        )
        netF = netF.to(accelerator.device)
        criterion_GAN = criterion_GAN.to(accelerator.device)
        for crit in nce_criteria:
            crit.to(accelerator.device)
        if mavic_criterion is not None:
            mavic_criterion = mavic_criterion.to(accelerator.device)
        if latent_target_encoder is not None:
            latent_target_encoder = latent_target_encoder.to(accelerator.device)
        if rep_alignment_module is not None:
            rep_alignment_module = rep_alignment_module.to(accelerator.device)

        if accelerator.is_main_process:
            project_name = f"cut-{cfg.task_name}"
            tracker_config = build_accelerate_tracker_config(cfg)
            tracker_init_kwargs = build_accelerate_tracker_init_kwargs(cfg, project_name)
            accelerator.init_trackers(
                project_name,
                config=tracker_config,
                init_kwargs=tracker_init_kwargs or {},
            )

        # Register schedulers for checkpointing (not passed to prepare)
        accelerator.register_for_checkpointing(scheduler_G, scheduler_D)

        # Hooks so accelerator.save_state/load_state use our diffusers-style layout
        def _save_model_hook(models, weights, output_dir):
            save_checkpoint_diffusers(
                output_dir,
                accelerator.unwrap_model(models[0]),
                scheduler=None,
                model_name="generator",
                pipeline_class_name="CUTPipeline",
                extra_state_dicts={
                    "discriminator": accelerator.unwrap_model(models[1]).state_dict(),
                    "feature_network": netF.state_dict(),
                },
            )

        def _load_model_hook(models, input_dir):
            self._load_cut_models_from_checkpoint(
                models[0], models[1], netF, input_dir, accelerator
            )

        accelerator.register_save_state_pre_hook(_save_model_hook)
        accelerator.register_load_state_pre_hook(_load_model_hook)

        global_step = 0
        first_epoch = 0
        # optimizer_F is created after data-dependent initialisation of netF
        # (see PatchSampleMLP.create_mlp which is called on first forward pass)
        optimizer_F = None

        # Resume
        if cfg.resume_from_checkpoint:
            path = cfg.resume_from_checkpoint
            if path == "latest":
                # Prefer step checkpoints (full state); fall back to latest epoch checkpoint (model-only)
                step_dirs = [
                    d for d in os.listdir(cfg.output_dir)
                    if d.startswith("checkpoint") and checkpoint_dir_sort_key(d)[0] == 0
                ]
                epoch_dirs = [
                    d for d in os.listdir(cfg.output_dir)
                    if d.startswith("checkpoint") and checkpoint_dir_sort_key(d)[0] == 1
                ]
                step_dirs = sorted(step_dirs, key=lambda x: checkpoint_dir_sort_key(x)[1])
                epoch_dirs = sorted(epoch_dirs, key=lambda x: checkpoint_dir_sort_key(x)[1])
                path = step_dirs[-1] if step_dirs else (epoch_dirs[-1] if epoch_dirs else None)
            if path is not None:
                if os.path.isabs(path) or os.path.sep in path:
                    load_path = os.path.abspath(path)
                else:
                    load_path = os.path.join(cfg.output_dir, path)
                if checkpoint_has_accelerator_state(load_path):
                    accelerator.load_state(load_path)
                    global_step = int(Path(path).name.split("-")[1])
                    first_epoch = global_step // num_update_steps_per_epoch
                    logger.info(f"Resumed from {path} (full state)")
                else:
                    # Model-only checkpoint (e.g. checkpoint-epoch-*): load weights, restart optimizers
                    self._load_cut_models_from_checkpoint(netG, netD, netF, load_path, accelerator)
                    # Infer step from path if checkpoint-{step}; else 0
                    parts = Path(path).name.split("-")
                    global_step = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
                    first_epoch = global_step // num_update_steps_per_epoch
                    logger.info(f"Resumed from {path} (model-only; optimizers reset)")

        num_epochs_this_run = total_epochs - first_epoch
        logger.info("***** Running CUT training *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Num examples     = {len(train_dataset)}")
        logger.info(f"  Num epochs       = {num_epochs_this_run}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {cfg.max_train_steps}")

        progress_bar = tqdm(
            range(global_step, cfg.max_train_steps),
            disable=not accelerator.is_local_main_process,
            desc=f"Training CUT {cfg.task_name}",
        )

        for epoch in range(first_epoch, total_epochs):
            netG.train()
            netD.train()

            for step, batch in enumerate(train_dataloader):
                real_A, real_B = self.preprocess_batch(batch, accelerator.device)
                pixel_real_A = real_A
                pixel_real_B = real_B
                if cfg.use_latent_target and latent_target_encoder is not None:
                    with torch.no_grad():
                        real_A = latent_target_encoder.encode(pixel_real_A)
                        real_B = latent_target_encoder.encode(pixel_real_B)

                # Data-dependent initialisation of netF (first step only).
                # PatchSampleMLP lazily creates its MLP layers on the first
                # forward pass based on the encoder feature dimensions;
                # optimizer_F is created once those parameters exist.
                if not netF.mlp_init:
                    with torch.no_grad():
                        fake_B_init = netG(real_A)
                        fake_B_for_enc = CUTTrainer._adapt_for_encoder(
                            fake_B_init, cfg.target_channels, cfg.source_channels
                        )
                        feat_init = netG(fake_B_for_enc, nce_layers, encode_only=True)
                        netF(feat_init, cfg.num_patches, None)
                    optimizer_F = create_optimizer(
                        netF.parameters(),
                        optimizer_type=cfg.optimizer_type,
                        lr=cfg.learning_rate,
                        betas=(cfg.beta1, cfg.beta2),
                        prodigy_d0=getattr(cfg, "prodigy_d0", 1e-6),
                    )

                with accelerator.accumulate(netG, netD):
                    # Forward G
                    fake_B = netG(real_A)

                    # ---- Update D ----
                    loss_D = self.compute_D_loss(netD, criterion_GAN, real_B, fake_B)
                    accelerator.backward(loss_D)
                    nan_skip = torch.isnan(loss_D) or torch.isinf(loss_D)
                    if not nan_skip:
                        accelerator.clip_grad_norm_(netD.parameters(), cfg.max_grad_norm)
                        optimizer_D.step()
                    optimizer_D.zero_grad()

                    # ---- Update G + F ----
                    if nan_skip:
                        loss_G = torch.tensor(float("nan"), device=real_A.device)
                        loss_G_GAN = loss_G
                        loss_NCE = loss_G
                        loss_NCE_Y = loss_G
                        loss_extras = {"loss_mavic": None, "loss_latent": None, "loss_rep_alignment": None}
                    else:
                        lambda_repa = (
                            lambda_repa_cosine(
                                global_step,
                                cfg.lambda_rep_alignment,
                                cfg.lambda_rep_alignment_end,
                                cfg.lambda_rep_alignment_decay_steps,
                            )
                            if rep_alignment_module is not None and cfg.lambda_rep_alignment_decay_steps > 0
                            else cfg.lambda_rep_alignment
                        )
                        loss_G, loss_G_GAN, loss_NCE, loss_NCE_Y, loss_extras = self.compute_G_loss(
                            netG, netD, netF, criterion_GAN, nce_criteria,
                            real_A, fake_B, real_B,
                            nce_layers=nce_layers,
                            lambda_GAN=cfg.lambda_GAN,
                            lambda_NCE=cfg.lambda_NCE,
                            nce_idt=cfg.nce_idt,
                            num_patches=cfg.num_patches,
                            mavic_criterion=mavic_criterion,
                            mavic_loss_weight=cfg.mavic_loss_weight,
                            latent_target_encoder=latent_target_encoder,
                            lambda_latent=cfg.lambda_latent,
                            rep_alignment_module=rep_alignment_module,
                            lambda_rep_alignment=lambda_repa,
                            pixel_target=pixel_real_B if cfg.use_latent_target else None,
                            pixel_source=pixel_real_A if cfg.use_latent_target else None,
                            latent_decode_fn=latent_target_encoder.decode if cfg.use_latent_target else None,
                            in_latent_space=cfg.use_latent_target,
                            source_channels=cfg.source_channels,
                            target_channels=cfg.target_channels,
                        )
                        accelerator.backward(loss_G)
                        nan_skip = torch.isnan(loss_G) or torch.isinf(loss_G)
                        if not nan_skip:
                            accelerator.clip_grad_norm_(g_params, cfg.max_grad_norm)
                            optimizer_G.step()
                            if optimizer_F is not None:
                                accelerator.clip_grad_norm_(netF.parameters(), cfg.max_grad_norm)
                                optimizer_F.step()
                        optimizer_G.zero_grad(set_to_none=True)
                        if optimizer_F is not None:
                            optimizer_F.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if nan_skip:
                        logger.warning("NaN/Inf detected in loss, skipped optimizer step (step %d)", global_step)

                    logs = {
                        "loss_D": loss_D.detach().item(),
                        "loss_G": loss_G.detach().item(),
                        "loss_G_GAN": loss_G_GAN.detach().item() if torch.is_tensor(loss_G_GAN) else loss_G_GAN,
                        "loss_NCE": loss_NCE.detach().item() if torch.is_tensor(loss_NCE) else loss_NCE,
                        "lr": optimizer_G.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                    if nan_skip:
                        logs["nan_skip"] = 1
                    if loss_extras.get("loss_rep_alignment") is not None:
                        logs["loss/repa"] = loss_extras["loss_rep_alignment"].item()
                        if cfg.lambda_rep_alignment_decay_steps > 0:
                            logs["lambda/repa"] = lambda_repa
                    if loss_extras.get("loss_mavic") is not None:
                        logs["loss/mavic"] = loss_extras["loss_mavic"].item()
                    if loss_extras.get("loss_latent") is not None:
                        logs["loss/latent"] = loss_extras["loss_latent"].item()
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # Step-based validation (main process only; TODO: add barrier when multi-GPU sync fixed)
                    if (
                        val_dataloader is not None
                        and cfg.validation_steps is not None
                        and global_step % cfg.validation_steps == 0
                        and accelerator.is_main_process
                    ):
                        val_result = self.log_validation(
                            netG, val_dataloader, accelerator, global_step,
                            latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                        )
                        if val_result:
                            accelerator.log(val_result, step=global_step)

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        # Full state (models + optimizer + scheduler) for resume
                        accelerator.save_state(save_path)
                        save_training_config(cfg, save_path)
                        logger.info(f"Saved state to {save_path}")
                        if cfg.push_to_hub and cfg.hub_model_id:
                            hub_subpath = f"cut/{cfg.task_name}"
                            if cfg.hub_path_tier:
                                hub_subpath = f"{hub_subpath}/{cfg.hub_path_tier}"
                            push_checkpoint_to_hub(
                                save_path,
                                hub_model_id=cfg.hub_model_id,
                                commit_message=f"cut {cfg.task_name} step {global_step}",
                                path_in_repo=f"{hub_subpath}/checkpoint-{global_step}",
                                request_timeout=30,
                            )

                        if cfg.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                                key=checkpoint_dir_sort_key,
                            )
                            for old in ckpts[: -cfg.checkpoints_total_limit]:
                                shutil.rmtree(os.path.join(cfg.output_dir, old))

                if global_step >= cfg.max_train_steps:
                    break

            # Step LR schedulers at epoch boundary
            if cfg.lr_policy == "linear":
                scheduler_G.step()
                scheduler_D.step()

            # Epoch-based validation (main process only; TODO: add barrier when multi-GPU sync fixed)
            if (
                val_dataloader is not None
                and cfg.validation_epochs is not None
                and (epoch + 1) % cfg.validation_epochs == 0
                and accelerator.is_main_process
            ):
                val_result = self.log_validation(
                    netG, val_dataloader, accelerator, global_step,
                    latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                )
                if val_result:
                    accelerator.log(val_result, step=global_step)

            # Save at epoch boundary
            if (
                accelerator.is_main_process
                and save_model_epochs is not None
                and (epoch + 1) % save_model_epochs == 0
            ):
                unwrapped_G = accelerator.unwrap_model(netG)
                unwrapped_D = accelerator.unwrap_model(netD)
                epoch_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch + 1}")
                save_checkpoint_diffusers(
                    epoch_dir,
                    unwrapped_G,
                    scheduler=None,
                    model_name="generator",
                    pipeline_class_name="CUTPipeline",
                    extra_state_dicts={
                        "discriminator": unwrapped_D.state_dict(),
                        "feature_network": netF.state_dict(),
                    },
                )
                save_training_config(cfg, epoch_dir)
                logger.info(f"Saved models at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    hub_subpath = f"cut/{cfg.task_name}"
                    if cfg.hub_path_tier:
                        hub_subpath = f"{hub_subpath}/{cfg.hub_path_tier}"
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"cut {cfg.task_name} epoch {epoch + 1}",
                        path_in_repo=f"{hub_subpath}/checkpoint-epoch-{epoch + 1}",
                        request_timeout=30,
                    )

                if cfg.checkpoints_total_limit is not None:
                    ckpts = sorted(
                        [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                        key=checkpoint_dir_sort_key,
                    )
                    for old in ckpts[: -cfg.checkpoints_total_limit]:
                        shutil.rmtree(os.path.join(cfg.output_dir, old))

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] CUT training complete!")
