import json
import logging
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from snp_transformer.data_objects import Individual
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.registry import Registry
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .positional_embeddings import PositionalEncodingModule

logger = logging.getLogger(__name__)


@dataclass
class InputIds:
    """

    Attributes:
        type_indexes: Indexes of the domain (e.g. snps and phenotype types such as height, weight)
        value_indexes: Indexes of the values (e.g. snp values, phenotype values)
        is_padding: Boolean tensor indicating which values are padding
    """

    domain_ids: torch.Tensor
    snp_value_ids: torch.Tensor
    snp_position_ids: torch.Tensor
    phenotype_value_ids: torch.Tensor
    phenotype_type_ids: torch.Tensor
    is_padding: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Validated the dataclass. Ensures that phenotype_value_ids are not all -1 and that snp_value_ids are not all -1
        """
        if torch.all(self.phenotype_value_ids == -1):
            raise ValueError("All phenotype values are -1")
        if torch.all(self.snp_value_ids == -1):
            raise ValueError("All snp values are -1")

        shape = self.domain_ids.shape
        for tensor in [
            self.domain_ids,
            self.snp_value_ids,
            self.snp_position_ids,
            self.phenotype_value_ids,
            self.phenotype_type_ids,
            self.is_padding,
        ]:
            assert tensor.shape == shape, f"Shape mismatch: {tensor.shape} != {shape}"

    def get_batch_size(self) -> int:
        return self.is_padding.shape[0]

    def get_max_seq_len(self) -> int:
        return self.is_padding.shape[1]

    def get_device(self) -> torch.device:
        return self.is_padding.device


@dataclass
class Embeddings:
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
    phenotype_value2idx: dict[str, int]
    phenotype_type2idx: dict[str, int]
    domain2idx: dict[str, int]
    mask_token: str = "MASK"
    pad_token: str = "PAD"
    snp_token: str = "snp"
    phenotype_token: str = "phenotype"

    @property
    def vocab_size_phenotype_value(self) -> int:
        return len(self.phenotype_value2idx)

    @property
    def vocab_size_snps(self) -> int:
        return len(self.snp2idx)

    @property
    def vocab_size_phenotype_types(self) -> int:
        return len(self.phenotype_type2idx)

    @property
    def vocab_size_types(self) -> int:
        return len(self.domain2idx)

    @property
    def vocab_size(self) -> int:
        # as e.g. snp2idx and phenotype2idx can have overlapping values for the mask and pad tokens
        unique_values = set(
            list(self.domain2idx.values())
            + list(self.snp2idx.values())
            + list(self.phenotype_value2idx.values()),
        )
        return len(unique_values)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls: type["Vocab"], vocab_dict: dict[str, Any]) -> "Vocab":
        return cls(**vocab_dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Ensure that no values are overlapping
        """
        self.check_no_duplicate_idx(self.snp2idx)
        self.check_no_duplicate_idx(self.domain2idx)
        self.check_no_duplicate_idx(self.phenotype_value2idx)
        self.check_no_duplicate_idx(self.phenotype_type2idx)

    @staticmethod
    def check_no_duplicate_idx(value2idx: Mapping) -> None:
        """
        Check that no values are overlapping
        """
        values = set(value2idx.values())
        if len(values) != len(value2idx):
            raise ValueError("Duplicate values in vocab.")


class Embedder(nn.Module):
    """
    Interface for embedding modules
    """

    d_model: int
    vocab: Vocab
    is_fitted: bool

    def __init__(self, *args, **kwargs):  # noqa
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any) -> Embeddings:
        ...

    @abstractmethod
    def fit(self, individuals: list[Individual], *args: Any) -> None:
        ...

    @abstractmethod
    def collate_individuals(self, individual: list[Individual]) -> InputIds:
        ...

    @abstractmethod
    def to_disk(self, save_dir: Path) -> None:
        ...

    @classmethod
    def from_disk(cls: type["Embedder"], save_dir: Path) -> "Embedder":
        ...


class SNPEmbedder(Embedder):
    mask_token = "MASK"
    pad_token = "PAD"

    def __init__(
        self,
        d_model: int,
        dropout_prob: float,
        max_sequence_length: int,
        positonal_embedding: PositionalEncodingModule,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.dropout_prob = dropout_prob

        assert (
            d_model == positonal_embedding.d_model
        ), "d_model is not the same for positional embedding and embedder"
        self.positional_embedding = positonal_embedding

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
        self.vocab = vocab

        self.snp_embedding = nn.Embedding(
            num_embeddings=vocab.vocab_size_snps,
            embedding_dim=self.d_model,
            padding_idx=vocab.snp2idx[self.pad_token],
        )

        self.phenotype_value_embedding = nn.Embedding(
            num_embeddings=vocab.vocab_size_phenotype_value,
            embedding_dim=self.d_model,
            padding_idx=vocab.phenotype_value2idx[self.pad_token],
        )

        self.phenotype_type_embedding = nn.Embedding(
            num_embeddings=vocab.vocab_size_phenotype_types,
            embedding_dim=self.d_model,
            padding_idx=vocab.phenotype_type2idx[self.pad_token],
        )

        self.domain_embedding = nn.Embedding(
            num_embeddings=vocab.vocab_size_types,
            embedding_dim=self.d_model,
            padding_idx=vocab.domain2idx[self.pad_token],
        )

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

        # add the domain (e.g. snps, phenotype) embeddings
        embeddings += self.domain_embedding(inputs.domain_ids)
        # NOTE: that the embeddings are added but that tokens for different domains
        # are not influenced by eachother as for the respective tokens we simply
        # add the zero tensor
        embeddings += self.snp_embedding(inputs.snp_value_ids)
        # add the phenotype value embeddings only to the phenotype embeddings
        embeddings += self.phenotype_value_embedding(inputs.phenotype_value_ids)
        # add the phenotype type embeddings only to the phenotype embeddings
        embeddings += self.phenotype_type_embedding(inputs.phenotype_type_ids)

        # Only update the positional embeddings for the SNPs
        is_snps = inputs.domain_ids == self.vocab.domain2idx[self.vocab.snp_token]
        embeddings[is_snps] += self.positional_embedding(inputs.snp_position_ids)[
            is_snps
        ]

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
        snps = individual.snps
        snp_ids = [self.vocab.snp2idx[str(snp)] for snp in snps.values]
        snp_ids = torch.tensor(snp_ids, dtype=torch.long)
        snp_pos_ids = torch.tensor(snps.bp, dtype=torch.long)

        # Phenotypes Values
        phenotypes = list(individual.phenotype.values())
        phenotypes_ids = [
            self.vocab.phenotype_value2idx[str(phenotype)] for phenotype in phenotypes
        ]
        phenotype_ids = torch.tensor(phenotypes_ids, dtype=torch.long)

        # Add padding for SNPs and Phenotypes
        # we want a vector on the length of the concatenated snps and phenotypes
        # with the padding tokens at the beginning for snps
        # and at the end for phenotypes
        snp_padding_id = self.vocab.snp2idx[self.pad_token]
        snp_pos_padding_id = 0  # is ignored in the forward pass
        phenotype_padding_id = self.vocab.phenotype_value2idx[self.pad_token]

        snp_padding = torch.ones_like(phenotype_ids, dtype=torch.long) * snp_padding_id
        snp_pos_padding = (
            torch.ones_like(phenotype_ids, dtype=torch.long) * snp_pos_padding_id
        )
        phenotype_padding = (
            torch.ones_like(snp_ids, dtype=torch.long) * phenotype_padding_id
        )
        _snp_ids = torch.cat([snp_padding, snp_ids])
        snp_pos_ids = torch.cat([snp_pos_padding, snp_pos_ids])
        phenotype_ids = torch.cat([phenotype_ids, phenotype_padding])

        # Phenotype Types
        phenotype_types_ = [
            self.vocab.phenotype_type2idx[ptype] for ptype in individual.phenotype
        ]
        phenotype_types = torch.tensor(phenotype_types_, dtype=torch.long)

        phenotype_type_padding_id = self.vocab.phenotype_type2idx[self.pad_token]
        phenotype_type_padding = (
            torch.ones_like(snp_ids, dtype=torch.long) * phenotype_type_padding_id
        )
        phenotype_types = torch.cat([phenotype_types, phenotype_type_padding])

        # Domain
        snp_idx = self.vocab.domain2idx["snp"]
        pheno_idx = self.vocab.domain2idx["phenotype"]
        domain_ids_: list[int] = [pheno_idx] * len(phenotypes) + [snp_idx] * len(snps)
        domain_ids = torch.tensor(domain_ids_, dtype=torch.long)

        # Padding (for masking)
        is_padding = torch.zeros_like(domain_ids, dtype=torch.bool)

        inputs_ids = {
            "snp_value_ids": _snp_ids,
            "snp_position_ids": snp_pos_ids,
            "phenotype_value_ids": phenotype_ids,
            "phenotype_type_ids": phenotype_types,
            "domain_ids": domain_ids,
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
            domain_ids=padded_seqs["domain_ids"],
            snp_value_ids=padded_seqs["snp_value_ids"],
            snp_position_ids=padded_seqs["snp_position_ids"],
            phenotype_value_ids=padded_seqs["phenotype_value_ids"],
            phenotype_type_ids=padded_seqs["phenotype_type_ids"],
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

        output = {}
        for key in keys:
            if key == "is_padding":
                continue
            values = [p[key][:max_seq_len] for p in sequences]
            values_w_padding = pad_sequence(
                values,
                batch_first=True,
                padding_value=self.vocab.domain2idx[self.pad_token],
            )
            output[key] = values_w_padding

        is_padding = [p["is_padding"][:max_seq_len] for p in sequences]
        is_padding_w_padding = pad_sequence(
            is_padding,
            batch_first=True,
            padding_value=True,
        )
        output["is_padding"] = is_padding_w_padding
        return output

    def fit(
        self,
        individuals: list[Individual],
    ) -> None:
        # could also be estimated from the data but there is no need for that
        snp2idx = {self.pad_token: 0, self.mask_token: 1, "1": 2, "2": 3}

        domain2idx: dict[str, int] = {
            self.pad_token: 0,
            "snp": 1,
            "phenotype": 2,
        }
        phenotype_value2idx: dict[str, int] = {
            self.pad_token: 0,
            self.mask_token: 1,
        }
        phenotype_type2idx: dict[str, int] = {
            self.pad_token: 0,
        }

        for ind in individuals:
            for pheno_type, value in ind.phenotype.items():
                if pheno_type not in phenotype_type2idx:
                    phenotype_type2idx[pheno_type] = len(phenotype_type2idx)
                value_ = str(value)
                if value_ not in phenotype_value2idx:
                    phenotype_value2idx[value_] = len(phenotype_value2idx)

        vocab: Vocab = Vocab(
            snp2idx=snp2idx,
            domain2idx=domain2idx,
            phenotype_value2idx=phenotype_value2idx,
            phenotype_type2idx=phenotype_type2idx,
            snp_token="snp",
            phenotype_token="phenotype",
        )

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

        # save positonal embedding
        self.positional_embedding.to_disk(save_dir / "positional_embedding.pt")
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

        kwargs["positonal_embedding"] = PositionalEncodingModule.from_disk(
            save_dir / "positional_embedding.pt",
        )

        embedder = cls(**kwargs)
        embedder.initialize_embeddings_layers(vocab)
        embedder.load_state_dict(torch.load(save_dir / "embedder.pt"))

        return embedder


@Registry.embedders.register("snp_embedder")
def create_snp_embedder(
    d_model: int,
    dropout_prob: float,
    max_sequence_length: int,
    positional_embedding: PositionalEncodingModule,
    individuals: Union[list[Individual], IndividualsDataset],
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
        positonal_embedding=positional_embedding,
    )

    if isinstance(individuals, IndividualsDataset):
        individuals = individuals.get_individuals()
    emb.fit(individuals)

    if checkpoint_path is not None:
        emb.to_disk(checkpoint_path)

    return emb
