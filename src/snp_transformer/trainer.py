"""
"""

from torch import nn
import torch

def train(
        model: nn.Module
        train_datalaoder: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        log_interval: int,
    ) -> None:
    """
    The main training function
    """
    pass