import logging
from abc import abstractmethod
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from snp_transformer.data_objects import Individual

from ..embedders import Embedder

logger = logging.getLogger(__name__)


@dataclass
class Targets:
    snp_targets: torch.Tensor
    phenotype_targets: torch.Tensor
    is_snp_mask: torch.Tensor
    is_phenotype_mask: torch.Tensor


class TrainableModule(pl.LightningModule):
    """
    Interface for a trainable module
    """

    embedding_module: Embedder

    @abstractmethod
    def collate_fn(self, individual: list[Individual]) -> Targets:
        ...
