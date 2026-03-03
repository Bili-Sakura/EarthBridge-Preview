# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Image buffer for training, adapted from the CUT codebase.

This module implements an image buffer that stores previously generated
images, enabling discriminator updates using a history of generated images
rather than only the latest ones.

Adapted from ``vendor/CUT/util/image_pool.py``.
"""

import random

import torch


class ImagePool:
    """Image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.

    Parameters
    ----------
    pool_size : int
        The size of the image buffer.  If ``pool_size == 0``, no buffer
        will be created and images are returned as-is.
    """

    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images: list = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return an image from the pool.

        Parameters
        ----------
        images : torch.Tensor
            The latest generated images from the generator.

        Returns
        -------
        torch.Tensor
            Images from the buffer.

        By 50%, the buffer will return input images.
        By 50%, the buffer will return images previously stored in the
        buffer, and insert the current images to the buffer.
        """
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
