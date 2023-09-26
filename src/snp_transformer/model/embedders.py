import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from snp_transformer.data_objects import Individual
from snp_transformer.registry import Registry
from torch import nn
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


@dataclass
class InputIds:
    domain_embeddings: dict[str, torch.Tensor]  # domain -> embeddings ids
    is_padding: torch.Tensor

    def get_batch_size(self) -> int:
        return self.is_padding.shape[0]

    def get_max_seq_len(self) -> int:
        return self.is_padding.shape[1]

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.domain_embeddings[key]

    def get_device(self) -> torch.device:
        return self.is_padding.device


@dataclass
class Vocab:
    domains: dict[str, dict[str, Any]]
    mask: str = "MASK"
    pad: str = "PAD"

    def get_padding_idx(self, key: str) -> int:
        return self.domains[key][self.pad]

    def get_mask_idx(self, key: str) -> int:
        return self.domains[key][self.mask]

    def get_vocab_size(self, key: str) -> int:
        return len(self.domains[key])

    def get_mask_ids(self) -> dict[str, int]:
        return {
            domain: domain_vocab[self.mask]
            for domain, domain_vocab in self.domains.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": self.domains,
            "mask": self.mask,
            "pad": self.pad,
        }

    @classmethod
    def from_dict(cls: type["Vocab"], vocab_dict: dict[str, Any]) -> "Vocab":
        return cls(**vocab_dict)


@runtime_checkable
class Embedder(Protocol):
    """
    Interface for embedding modules
    """

    d_model: int
    vocab: Vocab
    is_fitted: bool

    def __init__(self, *args: Any) -> None:
        ...

    def forward(self, *args: Any) -> dict[str, torch.Tensor]:
        ...

    def fit(self, individuals: list[Individual], *args: Any) -> None:
        ...

    def __call__(self, *args: Any) -> dict[str, torch.Tensor]:
        ...

    def collate_individuals(self, individual: list[Individual]) -> InputIds:
        ...


class SNPEmbedder(nn.Module, Embedder):
    def __init__(
        self,
        d_model: int,
        dropout_prob: float,
        max_sequence_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.dropout_prob = dropout_prob

        self.is_fitted: bool = False

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def check_if_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use")

    def initialize_embeddings_layers(
        self,
        vocab: Vocab,
    ) -> None:
        self.not_embeddings = {"is_padding"}
        self.vocab = vocab

        n_unique_snps = vocab.get_vocab_size("snp")
        self.n_unique_snps = n_unique_snps
        snp_embedding = nn.Embedding(n_unique_snps, self.d_model)

        self.embeddings = nn.ModuleDict({"snp": snp_embedding})

        # TODO: Add additional embeddings once the tests run

        self.is_fitted = True

    def forward(self, inputs: InputIds) -> dict[str, torch.Tensor]:
        self.check_if_fitted()

        batch_size = inputs.get_batch_size()
        max_seq_len = inputs.get_max_seq_len()

        device = inputs.get_device()

        # start embeddings as a zero tensor
        embeddings = torch.zeros(
            (batch_size, max_seq_len, self.d_model),
            dtype=torch.float,
            device=device,
        )

        for key, embedding in self.embeddings.items():
            embeddings += embedding(inputs[key])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        output_embeddings = {
            "embeddings": embeddings,
            "is_padding": inputs.is_padding,
        }
        return output_embeddings

    def _collate_individual(self, individual: Individual) -> dict[str, torch.Tensor]:
        snps = individual.snps
        snp_values = snps.values
        # convert to idx tensor

        snp_ids = torch.tensor(snp_values, dtype=torch.long)
        is_padding = torch.zeros_like(snp_ids, dtype=torch.bool)

        inputs_ids = {"snp": snp_ids, "is_padding": is_padding}
        return inputs_ids

    def collate_individuals(self, individual: list[Individual]) -> InputIds:
        """
        Handles padding and indexing by converting each to an index tensor
        """
        self.check_if_fitted()

        inds: list[dict[str, torch.Tensor]] = [
            self._collate_individual(ind) for ind in individual
        ]
        padded_seqs = self._pad_sequences(inds)
        input_ids = padded_seqs.pop("is_padding")
        return InputIds(domain_embeddings=padded_seqs, is_padding=input_ids)

    def _pad_sequences(
        self,
        sequences: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        keys = list(sequences[0].keys())
        any_key = keys[0]

        max_seq_len = max([len(p[any_key]) for p in sequences])
        assert max_seq_len <= self.max_sequence_length

        padded_sequences: dict[str, torch.Tensor] = {}

        for domain in self.vocab.domains:
            pad_idx = self.vocab.get_padding_idx(domain)
            padded_sequences[domain] = pad_sequence(
                [p[domain] for p in sequences],
                batch_first=True,
                padding_value=pad_idx,
            )

        key = "is_padding"
        padded_sequences[key] = padded_sequences[key] = pad_sequence(
            [p[key] for p in sequences],
            batch_first=True,
            padding_value=True,
        )

        return padded_sequences

    def fit(
        self,
        individuals: list[Individual],  # noqa: ARG002
        add_mask_token: bool = True,
    ) -> None:
        # could also be estimated from the data but there is no need for that
        # and this ensure that 1 and 2 is masked as one would expect
        snp2idx = {"0": 0, "1": 1, "2": 2, "PAD": 3}
        if add_mask_token:
            snp2idx["MASK"] = 3

        vocab: Vocab = Vocab(domains={"snp": snp2idx}, mask="MASK", pad="PAD")

        self.initialize_embeddings_layers(vocab)

    def to_disk(self, save_dir: Path):
        """
        Save the (trained) embedding to disk
        """
        save_dir.mkdir(exist_ok=True, parents=True)

        kwargs = {
            "d_model": self.d_model,
            "dropout_prob": self.dropout_prob,
            "max_sequence_length": self.max_sequence_length,
        }

        config_path = save_dir / "embedder_config.json"
        with config_path.open("w") as fp:
            json.dump(kwargs, fp)

        vocab_path = save_dir / "vocab.json"
        with vocab_path.open("w") as fp:
            json.dump(self.vocab.to_dict(), fp)

        torch.save(self.state_dict(), save_dir / "embedder.pt")

    @classmethod
    def from_disk(cls: type["SNPEmbedder"], save_dir: Path) -> "SNPEmbedder":
        """
        Load the (trained) embedding from disk
        """
        assert save_dir.is_dir()

        config_path = save_dir / "embedder_config.json"
        with config_path.open() as fp:
            kwargs = json.load(fp)

        vocab_path = save_dir / "vocab.json"
        with vocab_path.open() as fp:
            vocab_dict = json.load(fp)

        vocab = Vocab.from_dict(vocab_dict)

        embedder = cls(**kwargs)
        embedder.initialize_embeddings_layers(vocab)
        embedder.load_state_dict(torch.load(save_dir / "embedder.pt"))

        return embedder


@Registry.embedders.register("snp_embedder")
def create_snp_embedder(
    d_model: int,
    dropout_prob: float,
    max_sequence_length: int,
    individuals: list[Individual] | None = None,
    checkpoint_path: Path | None = None,
) -> Embedder:
    should_load_ckpt = (
        checkpoint_path is not None
        and checkpoint_path.is_dir()
        and list(checkpoint_path.iterdir())  # is not empty
    )

    if should_load_ckpt:
        emb = SNPEmbedder.from_disk(checkpoint_path)  # type: ignore

        kwargs_match = (
            emb.d_model == d_model
            and emb.dropout_prob == dropout_prob
            and emb.max_sequence_length == max_sequence_length
        )
        if kwargs_match:
            logger.info(f"Loaded embedder from {checkpoint_path}")
            return emb
        logger.warn(
            "Embedder kwargs do not match checkpoint kwargs, ignoring checkpoint",
        )

    emb = SNPEmbedder(
        d_model=d_model,
        dropout_prob=dropout_prob,
        max_sequence_length=max_sequence_length,
    )

    if individuals is None:
        individuals = []
    emb.fit(individuals)

    if checkpoint_path is not None:
        emb.to_disk(checkpoint_path)

    return emb
