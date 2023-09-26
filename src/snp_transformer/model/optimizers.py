from typing import Any, Callable

import torch

from ..registry import Registry


@Registry.optimizers.register("adam")
def create_adam(lr: float) -> Callable[[Any], torch.optim.Optimizer]:
    def configure_optimizers(parameters) -> torch.optim.Optimizer:  # noqa: ANN001
        return torch.optim.Adam(parameters, lr=lr)

    return configure_optimizers
