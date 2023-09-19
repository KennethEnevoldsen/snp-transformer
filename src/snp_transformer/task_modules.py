from copy import copy
from dataclasses import dataclass
from typing import Callable

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from .embedders import Embedder, InputIds, Vocab
from .registries import OptimizerFn


@dataclass
class MaskingTargets:
    domain_targets: dict[str, torch.Tensor]  # domain -> target ids
    padding_idx: int


class EncoderForMaskedLM(pl.LightningModule):
    """A LM head wrapper for masked language modeling."""

    def __init__(
        self,
        embedding_module: Embedder,
        encoder_module: nn.TransformerEncoder,
        create_optimizer_fn: OptimizerFn,
        domains_to_mask: list[str] | None = None,
    ):
        super().__init__()
        self.initialize_model(embedding_module, encoder_module, domains_to_mask)

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.create_optimizer_fn = create_optimizer_fn

    def initialize_model(
        self,
        embedding_module: Embedder,
        encoder_module: nn.TransformerEncoder,
        domains_to_mask: list[str] | None = None,
    ) -> None:
        self.embedding_module = embedding_module
        self.encoder_module = encoder_module

        self.d_model = self.embedding_module.d_model
        vocab: Vocab = self.embedding_module.vocab
        self.mask_token_ids = vocab.get_mask_ids()
        self.domains_to_mask = (
            vocab.domains if domains_to_mask is None else domains_to_mask
        )

        self.mlm_heads: nn.ModuleDict = nn.ModuleDict(
            {
                domain: nn.Linear(self.d_model, vocab.get_vocab_size(domain))
                for domain in self.domains_to_mask
            }
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        inputs: InputIds,
        masked_lm_labels: MaskingTargets,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        output = self.embedding_module(inputs)
        embeddings = output["embeddings"]
        is_padding = output["is_padding"]
        encoded_individuals = self.encoder_module(
            embeddings, src_key_padding_mask=is_padding
        )

        logits = {
            domain: self.mlm_heads[domain](encoded_individuals)
            for domain in self.domains_to_mask
        }
        # compute loss pr. domain
        domain_losses = {
            domain: self.loss(
                logits[domain].view(-1, logits[domain].size(-1)),
                masked_lm_labels.domain_targets[domain].view(-1),
            )
            for domain in self.domains_to_mask
        }

        # compute total loss using torch
        # this assumed equal weighting of domains
        total_loss = torch.stack(list(domain_losses.values())).sum()

        return {
            "logits": logits,
            "Training loss": total_loss,
            "Domain Training Losses": domain_losses,
        }

    def mask(
        self,
        domain_ids_tensor: torch.Tensor,
        domain_vocab_size: int,
        mask_token_id: int,
        masking_prob: float = 0.15,
        replace_with_mask_prob: float = 0.8,
        replace_with_random_prob: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masking function for the task
        """
        masked_lm_labels = domain_ids_tensor.clone()

        # Mask 15 % of the tokens
        prob = torch.rand(domain_ids_tensor.shape)
        mask = prob < masking_prob

        masked_lm_labels[~mask] = -1  # -1 will be ignored in loss function

        prob /= masking_prob

        # 80% of the time, replace with [MASK] token
        mask[mask.clone()] = prob[mask] < replace_with_mask_prob
        domain_ids_tensor[mask] = mask_token_id

        # 10% of the time, replace with random token
        prob /= 0.8
        mask[mask.clone()] = prob[mask] < replace_with_random_prob
        domain_ids_tensor[mask] = torch.randint(
            0, domain_vocab_size - 1, mask.sum().shape
        )

        # -> rest 10% of the time, keep the original word
        return domain_ids_tensor, masked_lm_labels

    def masking_fn(
        self, padded_sequence_ids: InputIds
    ) -> tuple[InputIds, MaskingTargets]:
        domain_embeddings = copy(padded_sequence_ids.domain_embeddings)
        masked_labels: dict[str, torch.Tensor] = {}
        is_padding = padded_sequence_ids.is_padding
        # perform masking
        for domain in self.domains_to_mask:
            domain_tensor = domain_embeddings[domain]
            mask_token_id = self.mask_token_ids[domain]
            domain_vocab_size = self.embedding_module.vocab.get_vocab_size(domain)

            masked_sequence, masked_label = self.mask(
                domain_tensor,
                domain_vocab_size=domain_vocab_size,
                mask_token_id=mask_token_id,
            )
            # Set padding to -1 to ignore in loss
            masked_label[is_padding] = -1

            domain_embeddings[domain] = masked_sequence
            masked_labels[domain] = masked_label

        return (
            InputIds(domain_embeddings, is_padding),
            MaskingTargets(masked_labels, padding_idx=-1),
        )

    def collate_fn(self, individuals: list) -> tuple[InputIds, MaskingTargets]:
        """
        Takes a list of individuals and returns a dictionary of padded sequence ids.
        """
        padded_sequence_ids = self.embedding_module.collate_individuals(individuals)
        masked_sequence_ids, masked_labels = self.masking_fn(padded_sequence_ids)
        return masked_sequence_ids, masked_labels

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        self.log_training_step(output)
        return output["Training loss"]

    def log_training_step(self, output) -> None:
        dom_train_losses = output.pop("Domain Training Losses")
        for domain in dom_train_losses:
            self.log(f"Training Loss ({domain})", dom_train_losses[domain])
        self.log("Training Loss", output["Training loss"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.create_optimizer_fn(self.parameters())
