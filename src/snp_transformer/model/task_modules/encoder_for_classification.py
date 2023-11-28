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
        vocab = self.embedding_module.vocab
        is_padding = padded_sequence_ids.is_padding

    #     snp_mask_token_id = vocab.snp2idx[vocab.mask_token]
    #     snp_value_ids = copy(padded_sequence_ids.snp_value_ids)

    #     is_snp_mask = (
    #         padded_sequence_ids.domain_ids == vocab.domain2idx[vocab.snp_token]
    #     )

    #     snp_ids_masked, snp_masked_label = self.mask(
    #         snp_value_ids,
    #         domain_vocab_size=vocab.vocab_size_snps,
    #         mask_token_id=snp_mask_token_id,
    #         is_padding=is_padding,
    #     )

    #     if self.mask_phenotype:
    #         phenotype_mask_token_id = vocab.phenotype_value2idx[vocab.mask_token]
    #         pheno_value_ids = copy(padded_sequence_ids.phenotype_value_ids)
    #         pheno_ids_masked, pheno_masked_label = self.mask(
    #             pheno_value_ids,
    #             domain_vocab_size=vocab.vocab_size_phenotype_value,
    #             mask_token_id=phenotype_mask_token_id,
    #             is_padding=is_padding,
    #         )

    #     else:
    #         pheno_ids_masked = torch.clone(padded_sequence_ids.phenotype_value_ids)
    #         pheno_masked_label = torch.clone(padded_sequence_ids.phenotype_value_ids)

    #     # Make sure to ignore padding when calculating loss and token from other domains
    #     is_pheno_or_padding = ~is_snp_mask | is_padding
    #     is_snp_or_padding = is_snp_mask | is_padding
    #     snp_masked_label[is_padding] = self.ignore_index
    #     pheno_masked_label[is_snp_or_padding] = self.ignore_index

    #     masked_sequence_ids = InputIds(
    #         domain_ids=padded_sequence_ids.domain_ids,
    #         snp_value_ids=snp_ids_masked,
    #         snp_position_ids=padded_sequence_ids.snp_position_ids,
    #         phenotype_value_ids=pheno_ids_masked,
    #         phenotype_type_ids=padded_sequence_ids.phenotype_type_ids,
    #         is_padding=is_padding,
    #     )

    #     targets = Targets(
    #         snp_targets=snp_masked_label,
    #         phenotype_targets=pheno_masked_label,
    #         is_snp_mask=~is_pheno_or_padding,
    #         is_phenotype_mask=~is_snp_or_padding,
    #     )

    #     return masked_sequence_ids, targets

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
