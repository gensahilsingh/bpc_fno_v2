"""Cosine decay learning rate schedule with linear warmup.

Implements a two-phase LR policy:
1. **Warmup** (steps 0 .. warmup_steps-1): linear ramp from 0 to lr_init.
2. **Cosine decay** (steps warmup_steps .. total_steps): smooth cosine
   anneal from lr_init down to lr_final.

The schedule is implemented as a :class:`torch.optim.lr_scheduler.LambdaLR`
subclass so it integrates directly with PyTorch Lightning's automatic LR
scheduler stepping.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWarmupScheduler(LambdaLR):
    """Cosine decay with linear warmup, compatible with Lightning.

    The optimizer's base LR is set to ``lr_init`` on construction so that
    the lambda multiplier can normalise everything relative to 1.0.

    Parameters
    ----------
    optimizer:
        Wrapped optimizer.
    warmup_steps:
        Number of steps for the linear warmup phase.
    total_steps:
        Total number of training steps (warmup + decay).
    lr_init:
        Peak learning rate reached at the end of warmup.
    lr_final:
        Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_init: float,
        lr_final: float,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_init = lr_init
        self.lr_final = lr_final

        # Set optimizer base LRs to lr_init so the lambda is a multiplier in [0, 1].
        for group in optimizer.param_groups:
            group["initial_lr"] = lr_init

        super().__init__(optimizer, lr_lambda=self._lr_lambda, last_epoch=-1)

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def get_lr_at_step(self, step: int) -> float:
        """Compute the learning rate for an arbitrary step.

        This is a convenience method that does not mutate scheduler state.

        Parameters
        ----------
        step:
            Training step (0-indexed).

        Returns
        -------
        float
            The learning rate at *step*.
        """
        return self.lr_init * self._lr_lambda(step)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lr_lambda(self, step: int) -> float:
        """Return a multiplicative factor in [lr_final/lr_init, 1.0]."""
        if step < self.warmup_steps:
            # Linear warmup: 0 -> 1 over warmup_steps
            return step / max(self.warmup_steps, 1)

        if step >= self.total_steps:
            return self.lr_final / self.lr_init if self.lr_init > 0 else 0.0

        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / max(decay_steps, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Interpolate between lr_init (factor=1) and lr_final (factor=lr_final/lr_init)
        lr_ratio = self.lr_final / self.lr_init if self.lr_init > 0 else 0.0
        return lr_ratio + (1.0 - lr_ratio) * cosine_factor
