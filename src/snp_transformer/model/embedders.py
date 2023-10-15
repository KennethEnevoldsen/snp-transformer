import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union, runtime_checkable

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from snp_transformer.data_objects import Individual
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.registry import Registry

from ..dataclass import DataclassAsDict

logger = logging.getLogger(__name__)


@dataclass
class InputIds(DataclassAsDict):
    """

    Attributes:
        type_indexes: Indexes of the domain (e.g. snps and phenotype types such as height, weight)
        value_indexes: Indexes of the values (e.g. snp values, phenotype values)
        is_padding: Boolean tensor indicating which values are padding
    """

    type_indexes: torch.Tensor
    value_indexes: torch.Tensor
    is_padding: torch.Tensor

    def get_batch_size(self) -> int:
        return self.is_padding.shape[0]

    def get_max_seq_len(self) -> int:
        return self.is_padding.shape[1]

    def get_device(self) -> torch.device:
        return self.is_padding.device


@dataclass
class Embeddings(DataclassAsDict):
    embeddings: torch.Tensor
    is_padding: torch.Tensor


@dataclass
class Vocab:
    """

    Attributes:
        snp2idx: Mapping from SNP value to index
        type2idx: Mapping from type to index. Type includes phenotype types (e.g. height, weight) and snp
        phenotype2idx: Mapping from phenotype value to index

    """

    snp2idx: dict[str, int]
    type2idx: dict[str, int]
    phenotype2idx: dict[str, int]
    mask: str = "MASK"
    pad: str = "PAD"

    @property
    def vocab_size_phenotype(self) -> int:
        return len(self.phenotype2idx)

    @property
    def vocab_size_snps(self) -> int:
        return len(self.snp2idx)

    @property
    def vocab_size_types(self) -> int:
        return len(self.type2idx)

    @property
    def vocab_size(self) -> int:
        # as e.g. snp2idx and phenotype2idx can have overlapping values for the mask and pad tokens
        unique_values = set(
            list(self.type2idx.values())
            + list(self.snp2idx.values())
            + list(self.phenotype2idx.values())
        )
        return len(unique_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snp2idx": self.snp2idx,
            "type2idx": self.type2idx,
            "phenotype2idx": self.phenotype2idx,
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
    mask_token = "MASK"
    pad_token = "PAD"

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

        # includes all embeddings
        self.embedding = nn.Embedding(vocab.vocab_size, self.d_model)

        self.is_fitted = True

    def forward(self, inputs: InputIds) -> Embeddings:
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

        # add the domain (e.g. snps, phenotype_type) embeddings
        embeddings += self.embedding(inputs.type_indexes)
        # add the value embeddings
        embeddings += self.embedding(inputs.value_indexes)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return Embeddings(
            embeddings=embeddings,
            is_padding=inputs.is_padding,
        )

    def _collate_individual(self, individual: Individual) -> dict[str, torch.Tensor]:
        """
        Collate an indivivual into ids
        """
        # SNP Values
        snps = individual.snps.values
        snp_ids = torch.tensor(snps, dtype=torch.long)

        # Phenotypes Values
        phenotypes = list(individual.phenotype.values())
        phenotype_ids = torch.tensor(phenotypes, dtype=torch.long)

        values_ids = torch.cat([snp_ids, phenotype_ids])

        # Type (e.g. snp and phenotypes such as height, weight)
        snp_type_idx = self.vocab.type2idx["snp"]
        snp_type_ids_: list[int] = [snp_type_idx] * len(snps)

        phenotype_types = list(individual.phenotype.keys())
        phenotype_type_ids_: list[int] = [
            self.vocab.type2idx[ptype] for ptype in phenotype_types
        ]

        type_ids = torch.tensor(snp_type_ids_ + phenotype_type_ids_, dtype=torch.long)

        # Padding
        is_padding = torch.zeros_like(values_ids, dtype=torch.bool)

        inputs_ids = {
            "values": values_ids,
            "type": type_ids,
            "is_padding": is_padding,
        }
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
        return InputIds(
            type_indexes=padded_seqs["type"],
            value_indexes=padded_seqs["values"],
            is_padding=padded_seqs["is_padding"],
        )

    def _pad_sequences(
        self,
        sequences: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        keys = list(sequences[0].keys())
        any_key = keys[0]

        max_seq_len = max([len(p[any_key]) for p in sequences])
        if max_seq_len <= self.max_sequence_length:
            logger.warning(
                "Sequence longer than max sequence length, truncating to max length",
            )
            max_seq_len = self.max_sequence_length

        values = [p["values"][:max_seq_len] for p in sequences]
        values_w_padding = pad_sequence(
            values,
            batch_first=True,
            padding_value=self.vocab.type2idx[self.pad_token],
        )

        types = [p["type"][:max_seq_len] for p in sequences]
        types_w_padding = pad_sequence(
            types,
            batch_first=True,
            padding_value=self.vocab.type2idx[self.pad_token],
        )

        is_padding = [p["is_padding"][:max_seq_len] for p in sequences]
        is_padding_w_padding = pad_sequence(
            is_padding,
            batch_first=True,
            padding_value=True,
        )
        return {
            "values": values_w_padding,
            "type": types_w_padding,
            "is_padding": is_padding_w_padding,
        }

    def fit(
        self,
        individuals: list[Individual],
    ) -> None:
        # could also be estimated from the data but there is no need for that
        # and this ensure that 1 and 2 is masked as one would expect
        snp2idx = {self.pad_token: 0, self.mask_token: 1, "0": 2, "1": 3, "2": 4}
        max_token = max(snp2idx.values())

        type2idx: dict[str, int] = {
            self.pad_token: 0,
            "snp": max_token + 1,
            "phenotype": max_token + 2,
        }
        max_token += 2
        phenotype2idx: dict[str, int] = {self.pad_token: 0, self.mask_token: 1}

        for ind in individuals:
            for pheno_type, value in ind.phenotype.items():
                if pheno_type not in type2idx:
                    # continue from max_token
                    max_token += 1
                    type2idx[pheno_type] = max_token
                if value not in phenotype2idx:
                    max_token += 1
                    phenotype2idx[str(value)] = max_token

        vocab: Vocab = Vocab(
            snp2idx=snp2idx,
            type2idx=type2idx,
            phenotype2idx=phenotype2idx,
        )

        assert vocab.vocab_size == max_token + 1

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
    individuals: Union[list[Individual], IndividualsDataset, None] = None,
    checkpoint_path: Union[Path, None] = None,
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
        logger.warning(
            "Embedder kwargs do not match checkpoint kwargs, ignoring checkpoint",
        )

    emb = SNPEmbedder(
        d_model=d_model,
        dropout_prob=dropout_prob,
        max_sequence_length=max_sequence_length,
    )

    if individuals is None:
        individuals = []
    if isinstance(individuals, IndividualsDataset):
        individuals = individuals.get_individuals()
    emb.fit(individuals)

    if checkpoint_path is not None:
        emb.to_disk(checkpoint_path)

    return emb
