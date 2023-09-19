"""
A simple transformer encoder for SNPs.
"""

import torch
from torch import nn

from snp_transformer.data_objects import Individual
from snp_transformer.embedders import Embedder


class Model(nn.Module):
    def __init__(
        self,
        snp_embedding_module: Embedder,
        encoder=nn.Module,
    ) -> None:
        # TODO add this assertion check to constructor!
        # assert (
        #     snp_embedding_module.embedding_dim == encoder_layer.embedding_dim
        # ), "Embedding dimensions must be equal"

        self.snp_embedding_module = snp_embedding_module
        self.encoder = encoder

    def forward(self, individuals: list[Individual]) -> torch.Tensor:
        """
        Embed the SNPs
        """

        inputs_ids: dict[
            str, torch.Tensor
        ] = self.snp_embedding_module.collate_individuals(individuals)

        embeddings = self.snp_embedding_module(inputs_ids)
        padding_mask = inputs_ids["is_padding"]
        contextualized = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        return contextualized
