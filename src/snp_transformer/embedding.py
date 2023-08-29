from typing import Any, Protocol


import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from snp_transformer.data_objects import Individual


class Embedding(Protocol):
    """
    Interface for embedding modules
    """

    def __init__(self, *args: Any) -> None:
        ...

    def forward(self, *args: Any) -> torch.Tensor:
        ...

    def collate_fn(
        self, individuals: list[Individual]
    ) -> list[dict[str, torch.Tensor]]:
        ...

    def fit(self, individuals: list[Individual], *args: Any) -> None:
        ...

    def __call__(self, *args: Any) -> torch.Tensor:
        ...


class SNPEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout_prob: float,
        max_sequence_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        self.is_fitted: bool = False

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def check_if_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use")

    def initialize_embeddings_layers(
        self,
        n_unique_snps: int,
    ) -> None:
        self.n_unique_snps = n_unique_snps
        snp_embedding = nn.Embedding(n_unique_snps, self.d_model)

        self.embeddings = nn.ModuleDict({"snp": snp_embedding})

        # TODO: Add additional embeddings once the tests run

        self.is_fitted = True

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.check_if_fitted()

        batch_size = inputs["snp"].shape[0]
        max_seq_len = inputs["snp"].shape[1]
        # start embeddings as a zero tensor
        embeddings = torch.zeros(
            (batch_size, max_seq_len, self.d_model),
            dtype=torch.float,
        )

        for key, embedding in self.embeddings.items():
            embeddings += embedding(inputs[key])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        output_embeddings = {
            "embeddings": embeddings,
            "is_padding": inputs["is_padding"],
        }
        return output_embeddings

    def collate_individual(self, individual: Individual) -> dict[str, torch.Tensor]:
        snps = individual.snps
        snp_values = snps.values
        # convert to idx tensor

        snp_ids = torch.tensor(snp_values, dtype=torch.long)
        is_padding = torch.zeros_like(snp_ids, dtype=torch.bool)

        inputs_ids = {"snp": snp_ids, "is_padding": is_padding}
        return inputs_ids

    def collate_fn(self, individual: list[Individual]) -> dict[str, torch.Tensor]:
        """
        Handles padding and indexing by converting each to an index tensor
        """
        self.check_if_fitted()

        inds: list[dict[str, torch.Tensor]] = [
            self.collate_individual(ind) for ind in individual
        ]
        return self.pad_sequences(inds)

    def pad_sequences(
        self,
        sequences: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        keys = list(sequences[0].keys())
        any_key = keys[0]

        max_seq_len = max([len(p[any_key]) for p in sequences])
        assert max_seq_len <= self.max_sequence_length

        padded_sequences: dict[str, torch.Tensor] = {}
        for key in keys:
            pad_idx = self.vocab[key]["PAD"]

            padded_sequences[key] = pad_sequence(
                [p[key] for p in sequences], batch_first=True, padding_value=pad_idx
            )

        return padded_sequences

    def fit(self, individuals: list[Individual], add_mask_token: bool = True):
        # could also be estimated from the data but there is no need for that
        # and this ensure that 1 and 2 is masked as one would expect
        snp2idx = {"0": 0, "1": 1, "2": 2, "PAD": 3}
        if add_mask_token:
            snp2idx["MASK"] = 3

        is_padding: dict[str, bool] = {"PAD": True}

        self.vocab: dict[str, dict[str, Any]] = {
            "snp": snp2idx,
            "is_padding": is_padding,
        }
        self.not_embeddings = {"is_padding"}
        n_unique_snps = len(snp2idx)

        self.initialize_embeddings_layers(
            n_unique_snps=n_unique_snps,
        )
