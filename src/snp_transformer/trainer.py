"""
"""

import torch
from torch import nn


def train(
    model: nn.Module,
    train_datalaoder: torch.utils.data.DataLoader,  # type: ignore
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    log_interval: int,
) -> None:
    """
    The main training function
    """
    ...
