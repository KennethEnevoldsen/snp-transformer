"""
A simple transformer encoder for SNPs.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        snp_embedding_module: nn.Module,
        encoder_layer=nn.Module,
        num_layers: int = 6,
    ) -> None:
        # TODO add this assertion check to constructor!
        # assert (
        #     snp_embedding_module.embedding_dim == encoder_layer.embedding_dim
        # ), "Embedding dimensions must be equal"

        self.snp_embedding_module = snp_embedding_module

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, individuals) -> torch.Tensor:
        """
        Embed the SNPs
        """
        batch_size = individuals.batch_size
        max_seq_len = len(individuals)

        x: torch.Tensor = self.snp_embedding_module(individuals.snps)
        assert x.shape[:2] == (batch_size, max_seq_len), "Shape is incorrect"
        embedding_shape = x.shape

        x = self.transformer_encoder(x)
        assert x.shape == (
            batch_size,
            max_seq_len,
            embedding_shape,
        ), "Shape of contextualized embeddings is incorrect"

        return x
