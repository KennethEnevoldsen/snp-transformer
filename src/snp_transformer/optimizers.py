from typing import Any, Callable

import torch

from .registries import optimizers


@optimizers.register("adam")
def create_adam(learning_rate: float) -> Callable[[Any], torch.optim.Optimizer]:
    def configure_optimizers(parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=learning_rate)

    return configure_optimizers
