"""
A dataset for loading in patients
"""

import logging
from pathlib import Path

from torch.utils.data import Dataset

from snp_transformer.data_objects import Individual, SNPs
from snp_transformer.dataset.loaders import Psparse, load_details, load_fam
from snp_transformer.registry import Registry

logger = logging.getLogger(__name__)


class IndividualsDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.fam_path = path.with_suffix(".fam")
        self.psparse_path = path.with_suffix(".psparse")
        self.details_path = path.with_suffix(".details")

        # ensure that they all exist
        error = f"does not exist in {path}, the following files exist: {list(path.glob('*'))}"
        assert self.fam_path.exists(), f"{self.fam_path} {error}"
        assert self.psparse_path.exists(), f"{self.psparse_path} {error}"
        assert self.details_path.exists(), f"{self.details_path} {error}"

        self.fam = load_fam(self.fam_path)
        self.snp_details = load_details(self.details_path)
        self.psparse = Psparse(self.psparse_path)
        self.idx2iid = self.fam.index.values

    def __len__(self) -> int:
        return self.fam.shape[0]

    def __getitem__(self, idx: int) -> Individual:
        iid = self.idx2iid[idx]
        ind = self.psparse[iid]

        snp_values = ind["Value"].values
        snp_indices = ind["SNP"].values

        snp_details = self.snp_details.iloc[snp_indices]  # type: ignore
        ind_fam = self.fam.loc[iid]

        snps = SNPs(
            values=list(snp_values),
            chromosomes=snp_details["chr"].values,
            cm=snp_details["cm"].values,
            bp=snp_details["bp"].values,
            a1=snp_details["a1"].values,
            a2=snp_details["a2"].values,
            gene=snp_details["gene"].values,
            exome=snp_details["exome"].values,
        )

        individual = Individual(
            snps=snps,
            iid=iid,
            fid=ind_fam.fid,
            father=ind_fam.father,
            mother=ind_fam.mother,
            sex=ind_fam.sex,
            phenotype={},
        )

        return individual


@Registry.datasets.register("individuals_dataset")
def create_individuals_dataset(path: Path) -> IndividualsDataset:
    logger.info("Creating dataset")
    return IndividualsDataset(path)
