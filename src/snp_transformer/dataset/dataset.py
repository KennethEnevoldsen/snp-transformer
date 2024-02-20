"""
A dataset for loading in patients
"""

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from torch.utils.data import Dataset, WeightedRandomSampler

from snp_transformer.data_objects import Individual, SNPs
from snp_transformer.dataset.loaders import (
    load_details,
    load_fam,
    load_pheno_folder,
    load_sparse,
)
from snp_transformer.registry import Registry

logger = logging.getLogger(__name__)


class IndividualsDataset(Dataset):
    def __init__(
        self,
        path: Path,
        split_path: Optional[Path] = None,
        pheno_dir: Optional[Path] = None,
        oversample_phenotypes: Optional[list[str]] = None,
        oversample_alpha: float = 1,
    ) -> None:
        """
        Args:
            path: path to the dataset
            split_path: optional path to a dataframe of individuals to use e.g. for splitting the dataset into train/val/test
            pheno_dir: Directory of phenotypes.
            oversample_phenotypes: List of phenotypes to oversample based on
            oversample_alpha: Hyperparameter for oversampling phenotypes. This follows the formula p(pheno) ∝ |pheno|^a, where |pheno| is the number
                of individuals with the phenotype label (0/1) and a is a hyperparameter. a=1 is equivalent to no oversampling and a=0 is equivalent to
                uniform sampling.
        """
        self.path = path
        self.fam_path = path.with_suffix(".fam")
        self.psparse_path = path.with_suffix(".sparse")
        self.details_path = path.with_suffix(".details")
        self.subset_path = split_path
        self.oversample_phenotypes = oversample_phenotypes
        self.oversample_alpha = oversample_alpha

        # ensure that they all exist
        error = f"does not exist in {path}, the following files exist: {[p.name for p in path.parent.glob('*')]}"
        assert self.fam_path.exists(), f"{self.fam_path} {error}"
        assert self.psparse_path.exists(), f"{self.psparse_path} {error}"
        assert self.details_path.exists(), f"{self.details_path} {error}"

        self.fam = load_fam(self.fam_path)
        self.snp_details = load_details(self.details_path)
        sparse = load_sparse(self.psparse_path)
        self.idx2iid = {i: str(iid) for i, iid in enumerate(self.fam.index.values)}

        self.iid2snp = self._sparse_to_iid2snp(sparse)
        pheno_folder = path.parent / "phenos" if pheno_dir is None else pheno_dir

        if pheno_folder.exists():
            self.iid2pheno = self.load_phenos(pheno_folder)
        else:
            self.iid2pheno = {str(iid): {} for iid in self.fam.index.values}
            logger.warning(f"No phenos folder found in {path.parent}")
        self.unique_phenos = {
            pheno for pheno_dict in self.iid2pheno.values() for pheno in pheno_dict
        }

        # Apply split if it exists
        self.all_iids = [str(iid) for iid in self.fam.index.values]
        self.valid_iids = self._read_split(self.subset_path)
        self.filter_individuals()

    def _sparse_to_iid2snp(self, sparse: pl.DataFrame) -> dict[str, pl.DataFrame]:
        idx2snp = sparse.partition_by("Individual", as_dict=True)
        iid2snp: dict[str, pl.DataFrame] = {}
        for idx, iid in self.idx2iid.items():
            iid2snp[iid] = idx2snp[idx + 1]
        return iid2snp

    def filter_individuals(self) -> None:
        self.idx2iid = dict(enumerate(self.valid_iids))

    def filter_phenotypes(self, phenotypes: Iterable[str]) -> None:
        """
        Filter individuals that does not have any of the specified phenotypes (logical OR)
        """
        valid_iids = set()
        valid_phenos = set(phenotypes)
        iid2pheno = self.iid2pheno
        for iid, pheno in iid2pheno.items():
            if valid_phenos.intersection(pheno):
                valid_iids.add(iid)

        self.valid_iids = valid_iids.intersection(self.valid_iids)
        self.filter_individuals()

    def _read_split(self, path: Optional[Path]) -> set[str]:
        if path is None:
            return set(self.all_iids)
        with path.open("r") as f:
            iids: list[str] = f.read().split("\n")
        return set(iids)

    def load_phenos(self, path: Path) -> dict[str, dict[str, int]]:
        pheno2iid = load_pheno_folder(path)
        iid2pheno: dict[str, dict[str, int]] = defaultdict(dict)
        for pheno, iid_map in pheno2iid.items():
            for iid, value in iid_map.items():
                iid2pheno[str(iid)][pheno] = value
        return iid2pheno

    def get_individuals(self) -> list[Individual]:
        return [self[i] for i in range(len(self))]

    def get_iids(self) -> list[str]:
        return [self.idx2iid[i] for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.idx2iid)

    def get_pheno(self, iid: str) -> dict[str, int]:
        if self.iid2pheno is None:
            return {}
        return self.iid2pheno[iid]

    def __getitem__(self, idx: int) -> Individual:
        iid = self.idx2iid[idx]
        ind = self.iid2snp[iid]
        phenos = self.get_pheno(iid)

        snp_values = ind["Value"].to_numpy()
        snp_indices = ind["SNP"].to_numpy() - 1  # sparse is 1-indexed

        snp_details = self.snp_details.iloc[snp_indices]  # type: ignore
        ind_fam = self.fam.loc[iid]

        snps = SNPs(
            values=list(snp_values),
            chromosomes=list(snp_details["chr"].values),
            cm=list(snp_details["cm"].values),
            bp=list(snp_details["bp"].values),
            a1=list(snp_details["a1"].values),
            a2=list(snp_details["a2"].values),
            gene=list(snp_details["gene"].values),
            exome=list(snp_details["exome"].values),
        )

        individual = Individual(
            snps=snps,
            iid=iid,
            fid=ind_fam.fid,
            father=ind_fam.father,
            mother=ind_fam.mother,
            sex=ind_fam.sex,
            phenotype=phenos,
        )

        return individual

    def train_test_split(
        self,
        prob: float = 0.8,
        train_path_prefix: str = "train",
        test_path_prefix: str = "test",
        seed: int = 42,
    ) -> None:
        """
        Splits the dataset into a train and test set
        """
        random.seed(seed)

        iids = list(self.idx2iid.values())
        random.shuffle(iids)
        split_idx = int(len(iids) * prob)
        train_iids = iids[:split_idx]
        test_iids = iids[split_idx:]

        train_path = self.path.parent / f"{train_path_prefix}.split"
        test_path = self.path.parent / f"{test_path_prefix}.split"

        self._write_split(train_path, train_iids)
        self._write_split(test_path, test_iids)

    def _write_split(self, path: Path, iids: list[str]) -> None:
        with path.open("w") as f:
            f.write("\n".join(iids))

    def create_weighted_sampler(self) -> Optional[WeightedRandomSampler]:
        if self.oversample_alpha == 1:
            return None
        if self.oversample_phenotypes is None:
            phenotypes = self.unique_phenos
            if len(phenotypes) == 0:
                return None
        else:
            phenotypes = self.oversample_phenotypes

        probs = []

        for pheno in phenotypes:
            phenos = [
                self.get_pheno(iid).get(pheno, None) for iid in self.idx2iid.values()
            ]
            label_freq = Counter(phenos)
            most_frequent_label = max(label_freq, key=label_freq.get)  # type: ignore
            # add None to most frequent label
            label_freq[most_frequent_label] += label_freq.pop(None, 0)
            # normalize
            label_freq_norm = {
                label: freq**self.oversample_alpha
                for label, freq in label_freq.items()
            }
            # prob for each of the labels
            total = sum(label_freq_norm.values())
            label2prob = {
                label: freq / total for label, freq in label_freq_norm.items()
            }
            label2indprob = {
                label: label2prob[label] / freq for label, freq in label_freq.items()
            }
            pheno_weights = np.array(
                [
                    label2indprob.get(pheno, label2indprob[most_frequent_label])
                    for pheno in phenos
                ],
            )

            assert (
                abs(np.sum(pheno_weights) - 1) < 1e-5
            ), f"Sum of weights is not 1: {np.sum(pheno_weights)}"
            probs.append(pheno_weights)

        # combine probabilities
        _probs = np.array(probs)
        _probs = np.prod(_probs, axis=0)
        # this might underflow, but we will deal with that if it happens

        # create sampler
        sampler = WeightedRandomSampler(_probs, len(_probs))
        return sampler


@Registry.datasets.register("individuals_dataset")
def create_individuals_dataset(
    path: Path,
    split_path: Optional[Path] = None,
    pheno_dir: Optional[Path] = None,
    oversample_phenotypes: Optional[list[str]] = None,
    oversample_alpha: float = 1,
) -> IndividualsDataset:
    logger.info("Creating dataset")

    return IndividualsDataset(
        path,
        split_path=split_path,
        pheno_dir=pheno_dir,
        oversample_phenotypes=oversample_phenotypes,
        oversample_alpha=oversample_alpha,
    )
