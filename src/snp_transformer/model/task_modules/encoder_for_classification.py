import logging
from copy import copy
from typing import Literal, Optional, Union

import torch
from snp_transformer.data_objects import Individual
from snp_transformer.dataset import IndividualsDataset
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from ...registry import OptimizerFn, Registry
from ..embedders import Embedder, InputIds, Vocab
from .encoder_for_masked_lm import EncoderForMaskedLM
from .trainable_modules import Targets, TrainableModule

logger = logging.getLogger(__name__)


class EncoderForClassification(TrainableModule):
    """A LM head wrapper for masked language modeling."""

    ignore_index = -1

    def __init__(
        self,
        phenotypes_to_predict: list[str],
        embedding_module: Embedder,
        encoder_module: nn.TransformerEncoder,
        create_optimizer_fn: OptimizerFn,
    ):
        super().__init__()
        self.phenotypes_to_predict = set(phenotypes_to_predict)
        self.save_hyperparameters()
        self.initialize_model(embedding_module, encoder_module)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.create_optimizer_fn = create_optimizer_fn
        self.initialize_metrics()

    def initialize_metrics(self):
        vocab: Vocab = self.embedding_module.vocab
        self.accuracy_pheno = MulticlassAccuracy(
            num_classes=vocab.vocab_size_phenotype_value,
            ignore_index=self.ignore_index,
        )

    def initialize_model(
        self,
        embedding_module: Embedder,
        encoder_module: nn.TransformerEncoder,
    ) -> None:
        self.embedding_module = embedding_module
        self.encoder_module = encoder_module

        self.d_model = self.embedding_module.d_model
        vocab: Vocab = self.embedding_module.vocab

        self.phenotype_head = nn.Linear(
            self.d_model,
            vocab.vocab_size_phenotype_value,
        )

        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    @classmethod
    def from_encoder_for_masked_lm(
        cls: type["EncoderForClassification"],
        phenotypes_to_predict: list[str],
        encoder_for_masked_lm: EncoderForMaskedLM,
        create_optimizer_fn: OptimizerFn,
    ) -> "EncoderForClassification":
        mdl = cls(
            phenotypes_to_predict=phenotypes_to_predict,
            embedding_module=encoder_for_masked_lm.embedding_module,
            encoder_module=encoder_for_masked_lm.encoder_module,
            create_optimizer_fn=create_optimizer_fn,
        )
        # check that heads are the same shape
        assert (
            mdl.phenotype_head.weight.shape
            == encoder_for_masked_lm.phenotype_head.weight.shape
        ), "The heads of the two models are not the same shape"

        # copy the phenotype head
        mdl.phenotype_head = copy(encoder_for_masked_lm.phenotype_head)
        return mdl

    def filter_dataset(self, dataset: IndividualsDataset) -> None:
        """
        Filter individuals that does not have the specified phenotypes
        """
        dataset.filter_phenotypes(self.phenotypes_to_predict)

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
            [vocab.phenotype_type2idx[pheno] for pheno in self.phenotypes_to_predict],
        )

        values_to_mask = torch.isin(
            padded_sequence_ids.phenotype_type_ids,
            pheno_ids_to_mask,
        )
        values = torch.where(
            values_to_mask,
            mask_id,
            padded_sequence_ids.phenotype_value_ids,
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
            self.phenotypes_to_predict.intersection(ind.phenotype)
            for ind in individuals
        ), "Not all individuals have the specified phenotypes"

        padded_sequence_ids = self.embedding_module.collate_individuals(individuals)
        masked_sequence_ids, masked_labels = self.mask_phenotypes(padded_sequence_ids)
        return masked_sequence_ids, masked_labels

    def forward(
        self,
        inputs: InputIds,
        targets: Targets,
    ) -> dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]:
        embeddings = self.embedding_module(inputs)

        encoded_individuals = self.encoder_module(
            embeddings.embeddings,
            src_key_padding_mask=embeddings.is_padding,
        )

        logits_pheno = self.phenotype_head(encoded_individuals)
        loss_pheno = self.loss(
            logits_pheno.view(-1, logits_pheno.size(-1)),
            targets.phenotype_targets.view(-1),
        )
        pheno_preds = torch.argmax(logits_pheno, dim=-1)
        pheno_acc = self.accuracy_pheno(pheno_preds, targets.phenotype_targets)

        if torch.isnan(loss_pheno):
            raise ValueError("Pheno loss is nan. This should not happen.")

        result = {
            "loss": loss_pheno,
            "Phenotype (all) Loss": loss_pheno,
            "Phenotype (all) Accuracy": pheno_acc,
        }

        result.update(self._metrics_pr_phenotype(targets, inputs, logits_pheno))

        return result

    def _metrics_pr_phenotype(
        self,
        targets: Targets,
        inputs: InputIds,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        computes the metrics (loss, and accuracy) for each phenotype
        """
        result = {}
        vocab = self.embedding_module.vocab
        for pheno in self.phenotypes_to_predict:
            pheno_idx: int = vocab.phenotype_type2idx[pheno]
            is_target_pheno = inputs.phenotype_type_ids == pheno_idx
            if torch.any(is_target_pheno):
                _logits = logits[is_target_pheno]
                _targets = targets.phenotype_targets[is_target_pheno]
                loss_pheno = self.loss(
                    _logits.view(-1, _logits.size(-1)),
                    _targets.view(-1),
                )
                result[f"Phenotype ({pheno}) Loss"] = loss_pheno

                pheno_preds = torch.argmax(_logits, dim=-1)
                pheno_acc = self.accuracy_pheno(pheno_preds, _targets)
                result[f"Phenotype ({pheno}) Accuracy"] = pheno_acc

        return result

    def training_step(
        self,
        batch: tuple[InputIds, Targets],
        batch_idx: int,  # noqa: ARG002
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        x, y = batch
        output = self.forward(x, y)
        self.log_step(output, batch_size=x.get_batch_size(), mode="Training")
        return output["loss"]

    def validation_step(
        self,
        batch: tuple[InputIds, Targets],
        batch_idx: int,  # noqa: ARG002
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        x, y = batch
        output = self.forward(x, y)

        self.log_step(output, x.get_batch_size(), mode="Validation")
        return output["loss"]

    def log_step(
        self,
        output: dict,
        batch_size: int,
        mode: Literal["Validation", "Training"],
    ) -> None:
        log_kwargs = {"batch_size": batch_size, "sync_dist": True}
        for key in output:
            self.log(f"{mode} {key}", output[key], **log_kwargs)


def get_phenotypes_to_predict(Embedder: Embedder) -> list[str]:
    vocab: Vocab = Embedder.vocab
    pred = []
    for pheno in vocab.phenotype_type2idx:
        if pheno != vocab.mask_token and pheno != vocab.pad_token:
            pred.append(pheno)
    return pred


@Registry.tasks.register("classification")
def create_classification_task(
    embedding_module: Embedder,
    encoder_module: nn.TransformerEncoder,
    create_optimizer_fn: OptimizerFn,
    phenotypes_to_predict: Optional[list[str]] = None,
) -> EncoderForClassification:
    if phenotypes_to_predict is None:
        pred_pheno = get_phenotypes_to_predict(embedding_module)
    else:
        pred_pheno = phenotypes_to_predict
    return EncoderForClassification(
        phenotypes_to_predict=pred_pheno,
        embedding_module=embedding_module,
        encoder_module=encoder_module,
        create_optimizer_fn=create_optimizer_fn,
    )


@Registry.tasks.register("classification_from_masked_lm")
def create_classification_task_from_masked_lm(
    encoder_for_masked_lm: EncoderForMaskedLM,
    create_optimizer_fn: OptimizerFn,
    phenotypes_to_predict: Optional[list[str]] = None,
) -> EncoderForClassification:
    if phenotypes_to_predict is None:
        pred_pheno = get_phenotypes_to_predict(encoder_for_masked_lm.embedding_module)
    else:
        pred_pheno = phenotypes_to_predict

    return EncoderForClassification.from_encoder_for_masked_lm(
        phenotypes_to_predict=pred_pheno,
        encoder_for_masked_lm=encoder_for_masked_lm,
        create_optimizer_fn=create_optimizer_fn,
    )
