"""
This code performs the embedding of the input data.
"""

import torch
from torch import nn


class SNPEmbedding(nn.Module):
    """
    This class implements the embedding of the input data.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_unique_snps: int = 2,
        n_unique_chromosomes: int = 24,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of the embedding. Defaults to 2.
            n_unique_snps: Number of unique SNPs in the input data. Defaults to 2. Indicating 1 and 2.
        """
        super().__init__()
        self.snp_embedding = nn.Embedding(n_unique_snps, embedding_dim)
        self.chromosome_embedding = nn.Embedding(n_unique_chromosomes, embedding_dim)

        self.positional_encoding = ...

        # TODO:
        # add gene embedding
        # add exome embedding
        # add chromosome embedding

        self.dropout = ...
        self.layer_norm = ...

    def forward(self, snp) -> torch.Tensor:
        """
        Forward pass of the embedding
        """
        x = self.snp_embedding(snp.values)
        x = x + self.chromosome_embedding(snp.chromosomes)
        x = x + self.positional_encoding(snp.positions)  # type: ignore

        x = self.dropout(x)  # type: ignore
        x = self.layer_norm(x)  # type: ignore
        return x
