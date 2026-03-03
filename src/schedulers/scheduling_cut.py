# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Learning-rate scheduling for CUT training.

CUT is a feed-forward GAN — it does not use iterative noise scheduling
like diffusion models.  Instead, the "scheduler" here manages the
learning-rate decay policy used during training.

The default CUT policy keeps the learning rate constant for the first
``n_epochs`` and then linearly decays it to zero over ``n_epochs_decay``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CUTSchedulerOutput:
    """Output class for the CUT scheduler.

    Attributes
    ----------
    lr : float
        Current learning rate after the step.
    """

    lr: float


class CUTScheduler:
    """Learning-rate scheduler for CUT training.

    Supports the following policies:

    * ``"linear"`` – Constant LR for *n_epochs*, then linearly decay to 0
      over *n_epochs_decay* additional epochs.
    * ``"step"`` – Multiply LR by ``gamma`` every ``step_size`` epochs.
    * ``"cosine"`` – Cosine annealing over total epochs.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimiser whose learning rate will be adjusted.
    lr_policy : str
        One of ``"linear"``, ``"step"``, or ``"cosine"``.
    n_epochs : int
        Number of epochs with the initial learning rate (used by
        ``"linear"`` and ``"cosine"``).
    n_epochs_decay : int
        Number of epochs to linearly decay the learning rate to zero
        (used by ``"linear"``).
    step_size : int
        Step size for the ``"step"`` policy.
    gamma : float
        Multiplicative factor for the ``"step"`` policy.
    last_epoch : int
        The index of the last epoch (for resuming).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_policy: str = "linear",
        n_epochs: int = 200,
        n_epochs_decay: int = 200,
        step_size: int = 50,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.lr_policy = lr_policy
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay

        if lr_policy == "linear":

            def lambda_rule(epoch):
                return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)

            self._scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)

        elif lr_policy == "step":
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch,
            )

        elif lr_policy == "cosine":
            total_epochs = n_epochs + n_epochs_decay
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, last_epoch=last_epoch,
            )

        else:
            raise NotImplementedError(f"LR policy [{lr_policy}] is not implemented")

    def step(self) -> CUTSchedulerOutput:
        """Advance one epoch and return the current learning rate."""
        self._scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        return CUTSchedulerOutput(lr=lr)

    def get_last_lr(self) -> float:
        """Return the last computed learning rate."""
        return self._scheduler.get_last_lr()[0]
