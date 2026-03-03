# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Simple helper functions adapted from the CUT codebase.

This module contains utility functions for image manipulation, tensor
conversion, file-system helpers, and other common operations used
throughout the CUT baseline.

Adapted from ``vendor/CUT/util/util.py``.
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image


def str2bool(v):
    """Parse a boolean value from a string."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        import argparse

        raise argparse.ArgumentTypeError("Boolean value expected.")


def copyconf(default_opt, **kwargs):
    """Copy a namespace and override specific fields."""
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def tensor2im(input_image, imtype=np.uint8):
    """Convert a Tensor array into a numpy image array.

    Parameters
    ----------
    input_image : tensor
        The input image tensor array, expected in [-1, 1].
    imtype : type
        The desired type of the converted numpy array.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk.

    Parameters
    ----------
    image_numpy : numpy array
        Input numpy array (H, W, C) in uint8.
    image_path : str
        The path of the image.
    aspect_ratio : float
        Aspect ratio for resizing.
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute gradients.

    Parameters
    ----------
    net : torch.nn.Module
        Torch network.
    name : str
        The name of the network.
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array."""
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """Create empty directories if they don't exist.

    Parameters
    ----------
    paths : str or list of str
        A list of directory paths.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """Create a single empty directory if it didn't exist.

    Parameters
    ----------
    path : str
        A single directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    """Resize label tensor using nearest-neighbor interpolation."""
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    """Resize image tensor using PIL."""
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i : i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)
