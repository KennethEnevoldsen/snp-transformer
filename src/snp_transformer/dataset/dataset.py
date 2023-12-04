"""
A dataset for loading in patients
"""

import logging
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

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
    def __init__(self, path: Path, split_path: Optional[Path] = None) -> None:
        """
        Args:
            path: path to the dataset
            split_path: optional path to a dataframe of individuals to use e.g. for splitting the dataset into train/val/test
        """
        self.path = path
        self.fam_path = path.with_suffix(".fam")
        self.psparse_path = path.with_suffix(".sparse")
        self.details_path = path.with_suffix(".details")
        self.subset_path = split_path

        # ensure that they all exist
        error = f"does not exist in {path}, the following files exist: {list(path.glob('*'))}"
        assert self.fam_path.exists(), f"{self.fam_path} {error}"
        assert self.psparse_path.exists(), f"{self.psparse_path} {error}"
        assert self.details_path.exists(), f"{self.details_path} {error}"

        self.fam = load_fam(self.fam_path)
        self.snp_details = load_details(self.details_path)
        sparse = load_sparse(self.psparse_path)
        self.idx2iid = {i: str(iid) for i, iid in enumerate(self.fam.index.values)}

        # Apply split if it exists
        self.all_iids = [str(iid) for iid in self.fam.index.values]
        self.valid_iids: list[str] = self._read_split(self.subset_path)
        self.filter_individuals()

        self.idx2snp = sparse.partition_by("Individual", as_dict=True)

        pheno_folder = path.parent / "phenos"
        if pheno_folder.exists():
            self.iid2pheno = self.load_phenos(pheno_folder)
        else:
            self.iid2pheno = {str(iid): {} for iid in self.fam.index.values}
            logger.warning(f"No phenos folder found in {path.parent}")

    def filter_individuals(self) -> None:
        self.idx2iid = {
            i: iid for i, iid in self.idx2iid.items() if iid in self.valid_iids
        }

    def filter_phenotypes(self, phenotypes: Iterable[str]) -> None:
        """
        Filter individuals that does not have the specified phenotypes
        """
        valid_iids = set()
        valid_phenos = set(phenotypes)
        iid2pheno = self.iid2pheno
        for iid, pheno in iid2pheno.items():
            if valid_phenos.intersection(pheno):
                valid_iids.add(iid)

        self.valid_iids = list(valid_iids)
        self.filter_individuals()

    def _read_split(self, path: Optional[Path]) -> list[str]:
        if path is None:
            return self.all_iids
        with path.open("r") as f:
            iids = f.read().split("\n")
        return iids

    def load_phenos(self, path: Path) -> dict[str, dict[str, int]]:
        pheno2iid = load_pheno_folder(path)
        iid2pheno: dict[str, dict[str, int]] = defaultdict(dict)
        for pheno, iid_map in pheno2iid.items():
            for iid, value in iid_map.items():
                iid2pheno[str(iid)][pheno] = value
        return iid2pheno

    def get_individuals(self) -> list[Individual]:
        return [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.idx2iid)

    def get_pheno(self, iid: str) -> dict[str, int]:
        if self.iid2pheno is None:
            return {}
        return self.iid2pheno[iid]

    def __getitem__(self, idx: int) -> Individual:
        iid = self.idx2iid[idx]
        ind = self.idx2snp[idx + 1]  # sparse is 1-indexed
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


@Registry.datasets.register("individuals_dataset")
def create_individuals_dataset(path: Path) -> IndividualsDataset:
    logger.info("Creating dataset")
    return IndividualsDataset(path)
