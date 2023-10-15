"""
A dataset for loading in patients
"""

import logging
from collections import defaultdict
from pathlib import Path

from snp_transformer.data_objects import Individual, SNPs
from snp_transformer.dataset.loaders import (
    load_details,
    load_fam,
    load_pheno_folder,
    load_sparse,
)
from snp_transformer.registry import Registry
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class IndividualsDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.fam_path = path.with_suffix(".fam")
        self.psparse_path = path.with_suffix(".sparse")
        self.details_path = path.with_suffix(".details")

        # ensure that they all exist
        error = f"does not exist in {path}, the following files exist: {list(path.glob('*'))}"
        assert self.fam_path.exists(), f"{self.fam_path} {error}"
        assert self.psparse_path.exists(), f"{self.psparse_path} {error}"
        assert self.details_path.exists(), f"{self.details_path} {error}"

        self.fam = load_fam(self.fam_path)
        self.snp_details = load_details(self.details_path)
        sparse = load_sparse(self.psparse_path)
        self.idx2iid = self.fam.index.values

        self.idx2snp = sparse.partition_by("Individual", as_dict=True)

        pheno_folder = path.parent / "pheno"
        if pheno_folder.exists():
            self.iid2pheno = self.load_phenos(pheno_folder)
        else:
            self.iid2pheno = None
            logger.warning(f"No pheno folder found in {path.parent}")

    def load_phenos(self, path: Path) -> dict[str, dict[str, int]]:
        pheno2iid = load_pheno_folder(path)
        iid2pheno: dict[str, dict[str, int]] = defaultdict(dict)
        for pheno, iid_map in pheno2iid.items():
            for iid, value in iid_map.items():
                iid2pheno[iid][pheno] = value
        return iid2pheno

    def get_individuals(self) -> list[Individual]:
        return [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        return self.fam.shape[0]

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


@Registry.datasets.register("individuals_dataset")
def create_individuals_dataset(path: Path) -> IndividualsDataset:
    logger.info("Creating dataset")
    return IndividualsDataset(path)
