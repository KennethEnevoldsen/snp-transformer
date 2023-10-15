import json
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from snp_transformer.data_objects import Individual
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.registry import Registry
from torch import nn
from torch.nn.utils.rnn import pad_sequence

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
    phenotype_value_ids: torch.Tensor
    phenotype_type_ids: torch.Tensor
    is_padding: torch.Tensor

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
    phenotype_value2idx: dict[Union[str, int], int]
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

        # Add padding for SNPs and Phenotypes
        # we want a vector on the length of the concatenated snps and phenotypes
        # with the padding tokens at the end for snps
        # and at the beginning for phenotypes
        snp_padding_id = self.vocab.snp2idx[self.pad_token]
        phenotype_padding_id = self.vocab.phenotype_value2idx[self.pad_token]

        snp_padding = torch.ones_like(phenotype_ids, dtype=torch.long) * snp_padding_id
        phenotype_padding = (
            torch.ones_like(snp_ids, dtype=torch.long) * phenotype_padding_id
        )
        snp_ids = torch.cat([snp_ids, snp_padding])
        phenotype_ids = torch.cat([phenotype_padding, phenotype_ids])

        # Phenotype Types
        phenotype_types_ = [
            self.vocab.domain2idx[ptype] for ptype in individual.phenotype
        ]
        phenotype_types = torch.tensor(phenotype_types_, dtype=torch.long)

        phenotype_type_padding_id = self.vocab.phenotype_type2idx[self.pad_token]
        phenotype_type_padding = (
            torch.ones_like(snp_ids, dtype=torch.long) * phenotype_type_padding_id
        )
        phenotype_types = torch.cat([phenotype_type_padding, phenotype_types])

        # Domain
        snp_idx = self.vocab.domain2idx["snp"]
        pheno_idx = self.vocab.domain2idx["phenotype"]
        domain_ids_: list[int] = [snp_idx] * len(snps) + [pheno_idx] * len(phenotypes)
        domain_ids = torch.tensor(domain_ids_, dtype=torch.long)

        # Padding (for masking)
        is_padding = torch.zeros_like(domain_ids, dtype=torch.bool)

        inputs_ids = {
            "snp_value_ids": snp_ids,
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
        phenotype_value2idx: dict[Union[int, str], int] = {
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
                if value not in phenotype_value2idx:
                    phenotype_value2idx[value] = len(phenotype_value2idx)

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
    return emb
