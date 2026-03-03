# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Representation alignment via pre-trained image encoders.

This module implements representation alignment losses inspired by REPA
(REPresentation Alignment, see ``vendor/REPA``).  The technique is
architecture-agnostic: a frozen pre-trained encoder extracts features from
the **target (ground truth)** image, while a trainable projection head maps
the translation model's output features into the same embedding space.  A
negative-cosine-similarity loss encourages the model to produce outputs
whose representations match the clean target as seen by the encoder.

.. important::

   Following the original REPA formulation, the **teacher** encoder must
   process the **target** image (the clean ground-truth), *not* the source
   input.  This means REPA is only applicable when a pre-trained encoder
   exists for the **target domain**:

   * **SAR → RGB** – ✅  Use ``MaRS-Base-RGB`` to encode the RGB target.
   * **SAR → EO** – ❌  No pre-trained EO encoder available.
   * **RGB → IR** – ❌  No pre-trained IR encoder available.
   * **SAR → IR** – ❌  No pre-trained IR encoder available.

Four concrete encoder strategies are provided (only ``MaRS-RGB`` is
currently applicable as a REPA teacher for the SAR→RGB task):

* **MaRS-RGB alignment** (default for SAR2RGB) – A frozen MaRS-RGB SwinV2
  image encoder extracts features from the **target RGB** image.  Loaded
  via ``transformers`` as a Swinv2Model.
  Checkpoint: ``models/BiliSakura/MaRS-Base-RGB``.

* **MaRS-SAR alignment** – A frozen MaRS-SAR SwinV2 image encoder.
  Checkpoint: ``models/BiliSakura/MaRS-Base-SAR``.
  ⚠️ Not suitable as a REPA teacher because no current task has SAR as its
  *target* domain.

* **SARCLIP alignment** – A frozen SARCLIP ViT-L/14 image encoder.
  Checkpoint: ``models/BiliSakura/SARCLIP-ViT-L-14``.
  ⚠️ Same limitation as MaRS-SAR.

* **DINOv3-sat alignment** – A frozen DINOv3-sat ViT-L encoder.
  Checkpoint: ``models/facebook/dinov3-vitl16-pretrain-sat493m``.
  ⚠️ Trained on general satellite imagery; may be used experimentally.

The alignment loss follows REPA's formulation (negative cosine similarity
averaged over the batch) and uses a 3-layer MLP projection head identical
to the one in ``vendor/REPA/models/sit.py::build_mlp``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build_projector(input_dim: int, projector_dim: int, output_dim: int) -> nn.Sequential:
    """Build a 3-layer MLP projection head."""
    return nn.Sequential(
        nn.Linear(input_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, output_dim),
    )


def normalize_to_01(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] range to [0, 1]."""
    return (images + 1.0) * 0.5


def adapt_channels(images: torch.Tensor) -> torch.Tensor:
    """Expand 1-channel images to 3 channels."""
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


class SARCLIPAlignment(nn.Module):
    """Representation alignment using SARCLIP encoder.
    
    Used for SAR2EO, SAR2IR, and SAR2RGB tasks.
    """

    def __init__(
        self,
        model_path: str = "./models/BiliSakura/SARCLIP-ViT-L-14",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from transformers import CLIPVisionModel, CLIPVisionModelWithProjection, CLIPImageProcessor
        
        # Load encoder (try with projection first)
        try:
            self.encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            self.encoder = CLIPVisionModel.from_pretrained(model_path, trust_remote_code=False)
        
        self.encoder.requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        
        # Get encoder_dim from model
        if encoder_dim is None:
            if hasattr(self.encoder.config, 'projection_dim'):
                self.encoder_dim = self.encoder.config.projection_dim
            elif hasattr(self.encoder, 'visual_projection'):
                self.encoder_dim = self.encoder.visual_projection.out_features
            else:
                self.encoder_dim = self.encoder.vision_model.config.hidden_size
        else:
            self.encoder_dim = encoder_dim
        
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None
        
        logger.info("Loaded SARCLIP encoder from %s (encoder_dim=%d, projector_dim=%d)", 
                   model_path, self.encoder_dim, self.projector_dim)

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SARCLIP."""
        import numpy as np
        from PIL import Image
        
        x = adapt_channels(normalize_to_01(images))
        pil_images = [Image.fromarray((x[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)) 
                      for i in range(x.shape[0])]
        return self.image_processor(pil_images, return_tensors="pt").pixel_values.to(images.device)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SARCLIP features from images."""
        x = self._preprocess(images).to(self.device)
        outputs = self.encoder(x)
        return outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs.last_hidden_state[:, 0]

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])  # global average pool
        if self.projector is not None:
            model_features = self.projector(model_features.float())
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


class DINOv3SatAlignment(nn.Module):
    """Representation alignment using DINOv3-sat encoder.
    
    Used for RGB2IR task.
    """

    def __init__(
        self,
        model_path: str = "./models/facebook/dinov3-vitl16-pretrain-sat493m",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        
        self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=False)
        self.encoder.requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.encoder_dim = encoder_dim or self.encoder.config.hidden_size
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None
        
        logger.info("Loaded DINOv3-sat encoder from %s (encoder_dim=%d, projector_dim=%d)", 
                   model_path, self.encoder_dim, self.projector_dim)

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv3."""
        import numpy as np
        from PIL import Image
        
        x = adapt_channels(normalize_to_01(images))
        pil_images = [Image.fromarray((x[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)) 
                      for i in range(x.shape[0])]
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        return inputs.pixel_values.to(images.device)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3-sat features from images."""
        x = self._preprocess(images).to(self.device)
        outputs = self.encoder(x)
        # Use pooler_output if available (as per official README), otherwise CLS token
        return outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])  # global average pool
        if self.projector is not None:
            model_features = self.projector(model_features.float())
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


class MaRSRGBAlignment(nn.Module):
    """Representation alignment using MaRS-RGB encoder.

    Default encoder for the RGB2IR task.  Uses ``transformers`` to load a
    pre-trained MaRS-RGB weights via ``AutoModel``.
    """

    def __init__(
        self,
        model_path: str = "./models/BiliSakura/MaRS-Base-RGB",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=False)
        self.encoder.requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)

        # Use config hidden_size or specified encoder_dim
        self.encoder_dim = encoder_dim or getattr(self.encoder.config, "hidden_size", 1024)
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None

        logger.info(
            "Loaded MaRS-RGB encoder from %s (encoder_dim=%d, projector_dim=%d)",
            model_path, self.encoder_dim, self.projector_dim,
        )

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract MaRS-RGB features (global-average-pooled)."""
        import numpy as np
        from PIL import Image

        x = adapt_channels(normalize_to_01(images))
        pil_images = [
            Image.fromarray((x[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))
            for i in range(x.shape[0])
        ]
        pixel_values = self.image_processor(pil_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        outputs = self.encoder(pixel_values)
        
        # Prefer pooler_output if available, otherwise GAP of last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
            
        last_hidden = outputs.last_hidden_state
        if last_hidden.ndim == 4:
            return last_hidden.mean(dim=[2, 3])  # (B, C, H, W) -> GAP
        return last_hidden.mean(dim=1)  # (B, L, C) -> token mean

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])
        if self.projector is not None:
            model_features = self.projector(model_features.float())
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


class MaRSSARAlignment(nn.Module):
    """Representation alignment using MaRS-SAR encoder.

    Default encoder for SAR2EO, SAR2IR, and SAR2RGB tasks.  Uses
    ``transformers`` to load a pre-trained MaRS-SAR weights via
    ``AutoModel``.
    """

    def __init__(
        self,
        model_path: str = "./models/BiliSakura/MaRS-Base-SAR",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=False)
        self.encoder.requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            do_convert_rgb=False,
        )

        # Use config hidden_size or specified encoder_dim
        self.encoder_dim = encoder_dim or getattr(self.encoder.config, "hidden_size", 1024)
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None

        logger.info(
            "Loaded MaRS-SAR encoder from %s (encoder_dim=%d, projector_dim=%d)",
            model_path, self.encoder_dim, self.projector_dim,
        )

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract MaRS-SAR features (global-average-pooled)."""
        import numpy as np
        from PIL import Image

        x = normalize_to_01(images)
        # Keep 1-channel for SAR; do NOT expand to 3-ch
        if x.shape[1] == 3:
            logger.warning("MaRS-SAR encoder received 3-channel input; using first channel only")
            x = x[:, :1]
        pil_images = [
            Image.fromarray((x[i, 0].cpu().float().numpy() * 255).astype(np.uint8), mode="L")
            for i in range(x.shape[0])
        ]
        pixel_values = self.image_processor(pil_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        outputs = self.encoder(pixel_values)
        
        # Prefer pooler_output if available, otherwise GAP of last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
            
        last_hidden = outputs.last_hidden_state
        if last_hidden.ndim == 4:
            return last_hidden.mean(dim=[2, 3])  # (B, C, H, W) -> GAP
        return last_hidden.mean(dim=1)  # (B, L, C) -> token mean

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])
        if self.projector is not None:
            model_features = self.projector(model_features.float())
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()
