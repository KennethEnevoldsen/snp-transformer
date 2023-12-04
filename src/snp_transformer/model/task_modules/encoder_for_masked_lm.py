import logging
from copy import copy
from typing import Literal, Union

import torch
from snp_transformer.dataset.dataset import IndividualsDataset
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from ...registry import OptimizerFn, Registry
from ..embedders import Embedder, InputIds, Vocab
from .trainable_modules import Targets, TrainableModule

logger = logging.getLogger(__name__)


class EncoderForMaskedLM(TrainableModule):
    """A LM head wrapper for masked language modeling."""

    ignore_index = -1

    def __init__(
        self,
        embedding_module: Embedder,
        encoder_module: nn.TransformerEncoder,
        create_optimizer_fn: OptimizerFn,
        mask_phenotype: bool = True,
    ):
        super().__init__()
        self.mask_phenotype = mask_phenotype
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
        self.accuracy_snp = MulticlassAccuracy(
            num_classes=vocab.vocab_size_snps,
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

        self.snp_head = nn.Linear(self.d_model, vocab.vocab_size_snps)

        if self.mask_phenotype:
            self.phenotype_head = nn.Linear(
                self.d_model,
                vocab.vocab_size_phenotype_value,
            )

        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

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

        logits_snp = self.snp_head(encoded_individuals)

        loss_snp = self.loss(
            logits_snp.view(-1, logits_snp.size(-1)),
            targets.snp_targets.view(-1),
        )

        snp_preds = torch.argmax(logits_snp, dim=-1)
        snp_acc = self.accuracy_snp(snp_preds, targets.snp_targets)

        result = {
            "loss": loss_snp,
            "SNP Loss": loss_snp,
            "SNP Accuracy": snp_acc,
        }

        if self.mask_phenotype:
            logits_pheno = self.phenotype_head(encoded_individuals)
            loss_pheno = self.loss(
                logits_pheno.view(-1, logits_pheno.size(-1)),
                targets.phenotype_targets.view(-1),
            )
            pheno_preds = torch.argmax(logits_pheno, dim=-1)
            pheno_acc = self.accuracy_pheno(pheno_preds, targets.phenotype_targets)

            # check if loss is nan (this happens when there are no phenotype values)
            if torch.isnan(loss_pheno):
                if torch.all(targets.is_phenotype_mask == False):  # noqa: E712
                    loss_pheno = torch.tensor(0.0, device=self.device)
                else:
                    raise ValueError("Pheno loss is nan")
            result["loss"] += loss_pheno  # assumes equal weighting of domains
            result["Phenotype Loss"] = loss_pheno
            result["Phenotype Accuracy"] = pheno_acc

        # check if loss is nan (this happens when there are no snps)
        if torch.isnan(result["loss"]):
            raise ValueError("Loss is nan")

        return result

    def mask(
        self,
        domain_ids_tensor: torch.Tensor,
        domain_vocab_size: int,
        mask_token_id: int,
        ignore_mask: torch.Tensor,
        masking_prob: float = 0.15,
        replace_with_mask_prob: float = 0.8,
        replace_with_random_prob: float = 0.1,
        ignore_index: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masking function for the task

        Args:
            domain_ids_tensor: A tensor of domain ids
            domain_vocab_size: The size of the domain vocabulary
            mask_token_id: The id of the mask token
            ignore_mask: A boolean tensor that indicates which tokens to ignore e.g. padding tokens.
            masking_prob: The probability of masking a token
            replace_with_mask_prob: The probability of replacing a token with the mask token
            replace_with_random_prob: The probability of replacing a token with a random token
            ignore_index: The index to ignore in the loss function
        """
        masked_lm_labels = domain_ids_tensor.clone()

        # Mask 15 % of the tokens
        prob = torch.rand(domain_ids_tensor.shape)
        mask = prob < masking_prob

        masked_lm_labels[~mask] = ignore_index  # will be ignored in loss function

        prob /= masking_prob

        # 80% of the time, replace with [MASK] token
        mask[mask.clone()] = prob[mask] < replace_with_mask_prob
        domain_ids_tensor[mask] = mask_token_id

        # 10% of the time, replace with random token
        prob /= 0.8
        mask[mask.clone()] = prob[mask] < replace_with_random_prob

        domain_ids_tensor[mask] = torch.randint(
            0,
            domain_vocab_size,
            mask.sum().shape,
        )

        # -> rest 10% of the time, keep the original word

        # ensure that at least one token is masked to avoid a nan loss
        no_pad_labels = masked_lm_labels[~ignore_mask]
        if torch.all(no_pad_labels == ignore_index):
            # mask a random non-padding token
            indices = torch.where(~ignore_mask)
            # select a random index form the indices
            idx = torch.randint(0, len(indices[0]), size=(1,))
            idx = tuple([i[idx] for i in indices])
            masked_lm_labels[idx] = domain_ids_tensor[idx]

        assert not torch.all(masked_lm_labels[~ignore_mask] == ignore_index)

        return domain_ids_tensor, masked_lm_labels

    def masking_fn(
        self,
        padded_sequence_ids: InputIds,
    ) -> tuple[InputIds, Targets]:
        vocab = self.embedding_module.vocab
        is_padding = padded_sequence_ids.is_padding

        snp_mask_token_id = vocab.snp2idx[vocab.mask_token]
        snp_value_ids = copy(padded_sequence_ids.snp_value_ids)

        is_snp_mask = (
            padded_sequence_ids.domain_ids == vocab.domain2idx[vocab.snp_token]
        )
        is_pheno_or_padding = ~is_snp_mask | is_padding
        is_snp_or_padding = is_snp_mask | is_padding

        snp_ids_masked, snp_masked_label = self.mask(
            snp_value_ids,
            domain_vocab_size=vocab.vocab_size_snps,
            mask_token_id=snp_mask_token_id,
            ignore_mask=is_pheno_or_padding,
        )

        if self.mask_phenotype:
            phenotype_mask_token_id = vocab.phenotype_value2idx[vocab.mask_token]
            pheno_value_ids = copy(padded_sequence_ids.phenotype_value_ids)
            pheno_ids_masked, pheno_masked_label = self.mask(
                pheno_value_ids,
                domain_vocab_size=vocab.vocab_size_phenotype_value,
                mask_token_id=phenotype_mask_token_id,
                ignore_mask=is_snp_or_padding,
            )

        else:
            pheno_ids_masked = torch.clone(padded_sequence_ids.phenotype_value_ids)
            pheno_masked_label = torch.clone(padded_sequence_ids.phenotype_value_ids)

        # Make sure to ignore padding when calculating loss and token from other domains
        snp_masked_label[is_padding] = self.ignore_index
        pheno_masked_label[is_snp_or_padding] = self.ignore_index

        masked_sequence_ids = InputIds(
            domain_ids=padded_sequence_ids.domain_ids,
            snp_value_ids=snp_ids_masked,
            snp_position_ids=padded_sequence_ids.snp_position_ids,
            phenotype_value_ids=pheno_ids_masked,
            phenotype_type_ids=padded_sequence_ids.phenotype_type_ids,
            is_padding=is_padding,
        )

        targets = Targets(
            snp_targets=snp_masked_label,
            phenotype_targets=pheno_masked_label,
            is_snp_mask=~is_pheno_or_padding,
            is_phenotype_mask=~is_snp_or_padding,
        )

        return masked_sequence_ids, targets

    def collate_fn(self, individuals: list) -> tuple[InputIds, Targets]:
        """
        Takes a list of individuals and returns a dictionary of padded sequence ids.
        """
        padded_sequence_ids = self.embedding_module.collate_individuals(individuals)
        masked_sequence_ids, masked_labels = self.masking_fn(padded_sequence_ids)
        return masked_sequence_ids, masked_labels

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

    def filter_dataset(self, dataset: IndividualsDataset) -> None:  # noqa: ARG002
        """
        placeholder function to filter dataset
        """
        return


@Registry.tasks.register("masked_lm")
def create_encoder_for_masked_lm(
    embedding_module: Embedder,
    encoder_module: nn.TransformerEncoder,
    create_optimizer_fn: OptimizerFn,
    mask_phenotype: bool,
) -> EncoderForMaskedLM:
    logger.info("Creating task module for masked lm")
    return EncoderForMaskedLM(
        embedding_module=embedding_module,
        encoder_module=encoder_module,
        create_optimizer_fn=create_optimizer_fn,
        mask_phenotype=mask_phenotype,
    )


@Registry.tasks.register("masked_lm_from_disk")
def create_encoder_for_masked_lm_from_disk(
    path: str,
) -> EncoderForMaskedLM:
    mdl = EncoderForMaskedLM.load_from_checkpoint(path)
    return mdl
