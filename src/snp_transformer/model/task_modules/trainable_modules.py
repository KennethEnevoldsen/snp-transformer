import logging
from abc import abstractmethod
from dataclasses import dataclass

import lightning.pytorch as pl
import torch

from snp_transformer.data_objects import Individual

from ..embedders import Embedder, InputIds

logger = logging.getLogger(__name__)


@dataclass
class MaskingTargets:
    domain_targets: dict[str, torch.Tensor]  # domain -> target ids
    padding_idx: int


class TrainableModule(pl.LightningModule):
    """
    Interface for a trainable module
    """

    embedding_module: Embedder

    @abstractmethod
    def collate_fn(self, individual: list[Individual]) -> InputIds:
        ...
