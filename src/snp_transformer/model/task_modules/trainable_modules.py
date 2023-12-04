import logging
from abc import abstractmethod
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from snp_transformer.data_objects import Individual
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.registry import OptimizerFn

from ..embedders import Embedder

logger = logging.getLogger(__name__)


@dataclass
class Targets:
    snp_targets: torch.Tensor
    phenotype_targets: torch.Tensor
    is_snp_mask: torch.Tensor
    is_phenotype_mask: torch.Tensor
    mask_id = -1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Ensures that some targets are masked abd
        """
        pheno_targets = self.phenotype_targets[self.is_phenotype_mask]
        if pheno_targets.numel() != 0 and torch.all(pheno_targets == self.mask_id):
            raise ValueError("No phenotype targets are masked")

        snp_targets = self.snp_targets[self.is_snp_mask]
        if torch.all(snp_targets == self.mask_id):
            raise ValueError("No SNP targets are masked")


class TrainableModule(pl.LightningModule):
    """
    Interface for a trainable module
    """

    embedding_module: Embedder
    create_optimizer_fn: OptimizerFn

    @abstractmethod
    def collate_fn(self, individual: list[Individual]) -> Targets:
        ...

    def filter_dataset(self, dataset: IndividualsDataset) -> None:
        """
        Filter individuals that does not have the specified requirements
        """
        ...

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.create_optimizer_fn(self.parameters())
