from functools import partial
from typing import Any, Callable

import torch

from ..registry import Registry
from collections.abc import Iterable
from functools import partial
from typing import Callable

import torch
from torch.optim.lr_scheduler import _LRScheduler 


LRSchedulerConfig = dict[str, Any]
LRSchedulerFn = Callable[[torch.optim.Optimizer], LRSchedulerConfig]

def configure_optimizers(parameters, lr) -> torch.optim.Optimizer:  # noqa: ANN001
    return torch.optim.Adam(parameters, lr=lr)


@Registry.optimizers.register("adam")
def create_adam(lr: float) -> Callable[[Any], torch.optim.Optimizer]:
    return partial(configure_optimizers, lr=lr)


@Registry.lr_schedulers.register("linear_schedule_with_warmup")
def create_linear_schedule_with_warmup(
    peak_lr: float = 0.01,
    num_training_steps: int =100_000,
    num_warmup_steps: int = 10_000,
) -> LRSchedulerFn:
    total_steps = num_training_steps
    warmup_steps = num_warmup_steps

    initial_lr = 1e-7  # Close to zero
    final_lr = 1e-7  # Close to zero


    def create_scheduler(optimizer: torch.optim.Optimizer) -> LRSchedulerConfig:
        # Setup the LinearLR schedulers for warmup and decay
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=initial_lr/peak_lr, end_factor=1.0, total_iters=warmup_steps)
        scheduler_decay = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=final_lr/peak_lr, total_iters=total_steps - warmup_steps)

        # Combine them using SequentialLR
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
        # return scheduler  # type: ignore
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "Validation loss",
            "strict": True,
            "name": None,
        }

    return create_scheduler