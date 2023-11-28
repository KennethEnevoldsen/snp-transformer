import logging
from copy import copy
from typing import Literal, Union

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from snp_transformer import IndividualsDataset
from snp_transformer.data_objects import Individual

from ...registry import OptimizerFn, Registry
from ..embedders import Embedder, InputIds, Vocab
from .encoder_for_masked_lm import EncoderForMaskedLM
from .trainable_modules import Targets, TrainableModule

logger = logging.getLogger(__name__)


class EncoderForClassification(EncoderForMaskedLM):
    """A LM head wrapper for masked language modeling."""

    ignore_index = -1

    def __init__(
        self,
        phenotypes: list[str],
        embedding_module: Embedder,
        encoder_module: nn.TransformerEncoder,
        create_optimizer_fn: OptimizerFn,
        mask_phenotype: bool = True,
    ):
        super().__init__(
            embedding_module=embedding_module,
            encoder_module=encoder_module,
            create_optimizer_fn=create_optimizer_fn,
            mask_phenotype=mask_phenotype,
        )
        self.phenotypes = set(phenotypes)

    @classmethod
    def from_encoder_for_masked_lm(
        cls,
        encoder_for_masked_lm: EncoderForMaskedLM,
    ) -> "EncoderForClassification":
        return cls(
            embedding_module=encoder_for_masked_lm.embedding_module,
            encoder_module=encoder_for_masked_lm.encoder_module,
            create_optimizer_fn=encoder_for_masked_lm.create_optimizer_fn,
            mask_phenotype=encoder_for_masked_lm.mask_phenotype,
        )

    def filter_dataset(self, dataset: IndividualsDataset) -> None:
        """
        Filter individuals that does not have the specified phenotypes
        """
        dataset.filter_phenotypes(self.phenotypes)

    def mask_phenotypes(
        self,
        padded_sequence_ids: InputIds,
    ) -> tuple[InputIds, Targets]:
        """
        mask phenotypes and return the masked sequence ids and the targets
        """
        vocab = self.embedding_module.vocab
        mask_id = vocab.phenotype_value2idx[vocab.mask_token]
        pheno_ids_to_mask = torch.tensor(
            [vocab.phenotype_type2idx[pheno] for pheno in self.phenotypes]
        )

        values_to_mask = torch.isin(
            padded_sequence_ids.phenotype_type_ids, pheno_ids_to_mask
        )
        values = torch.where(
            values_to_mask, mask_id, padded_sequence_ids.phenotype_value_ids
        )

        masked_sequence_ids = InputIds(
            domain_ids=padded_sequence_ids.domain_ids,
            snp_value_ids=padded_sequence_ids.snp_value_ids,
            snp_position_ids=padded_sequence_ids.snp_position_ids,
            phenotype_value_ids=values,
            phenotype_type_ids=padded_sequence_ids.phenotype_type_ids,
            is_padding=padded_sequence_ids.is_padding,
        )

        # creating targets
        is_padding = padded_sequence_ids.is_padding
        is_snp_mask = (
            padded_sequence_ids.domain_ids == vocab.domain2idx[vocab.snp_token]
        )
        is_pheno_or_padding = ~is_snp_mask | is_padding
        is_snp_or_padding = is_snp_mask | is_padding

        targets = Targets(
            snp_targets=padded_sequence_ids.snp_value_ids,
            phenotype_targets=padded_sequence_ids.phenotype_value_ids,
            is_snp_mask=~is_pheno_or_padding,
            is_phenotype_mask=~is_snp_or_padding,
        )
        return masked_sequence_ids, targets


    def collate_fn(self, individuals: list[Individual]) -> tuple[InputIds, Targets]:
        """
        Takes a list of individuals and returns a dictionary of padded sequence ids.
        """
        assert all(
            self.phenotypes.intersection(ind.phenotype) for ind in individuals
        ), "Not all individuals have the specified phenotypes"

        padded_sequence_ids = self.embedding_module.collate_individuals(individuals)
        masked_sequence_ids, masked_labels = self.mask_phenotypes(padded_sequence_ids)
        return masked_sequence_ids, masked_labels
