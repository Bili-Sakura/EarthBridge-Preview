# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Minimal FP16 utilities for UNet models.

This module provides simple conversion functions for mixed precision training.
For full mixed precision support, use accelerate's native capabilities.
"""

import torch.nn as nn


def convert_module_to_f16(module):
    """
    Convert primitive modules to float16.
    
    Args:
        module: A torch.nn module (typically Conv1d, Conv2d, or Conv3d).
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


def convert_module_to_f32(module):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    
    Args:
        module: A torch.nn module (typically Conv1d, Conv2d, or Conv3d).
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()
