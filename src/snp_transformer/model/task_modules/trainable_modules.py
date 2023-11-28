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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Ensures that not all targets are masked
        """
        pheno_targets = self.phenotype_targets[self.is_phenotype_mask]
        if pheno_targets.numel() != 0 and torch.all(pheno_targets == -1):
            raise ValueError("All phenotype targets are masked")

        snp_targets = self.snp_targets[self.is_snp_mask]
        if torch.all(snp_targets == -1):
            raise ValueError("All SNP targets are masked")


class TrainableModule(pl.LightningModule):
    """
    Interface for a trainable module
    """

    embedding_module: Embedder

    @abstractmethod
    def collate_fn(self, individual: list[Individual]) -> Targets:
        ...
