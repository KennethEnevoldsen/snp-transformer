from functools import partial
from typing import Any, Callable

import torch

from ..registry import Registry


def configure_optimizers(parameters, lr) -> torch.optim.Optimizer:  # noqa: ANN001
    return torch.optim.Adam(parameters, lr=lr)


@Registry.optimizers.register("adam")
def create_adam(lr: float) -> Callable[[Any], torch.optim.Optimizer]:
    return partial(configure_optimizers, lr=lr)
