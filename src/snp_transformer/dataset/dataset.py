"""
A dataset for loading in patients
"""

import logging
from pathlib import Path

from torch.utils.data import Dataset

from snp_transformer.data_objects import Individual, SNPs
from snp_transformer.dataset.loaders import load_details, load_fam, load_sparse
from snp_transformer.registry import Registry

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

    def __len__(self) -> int:
        return self.fam.shape[0]

    def __getitem__(self, idx: int) -> Individual:
        iid = self.idx2iid[idx]
        ind = self.idx2snp[idx]

        snp_values = ind["Value"].to_numpy()
        snp_indices = ind["SNP"].to_numpy()

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
            phenotype={},
        )

        return individual


@Registry.datasets.register("individuals_dataset")
def create_individuals_dataset(path: Path) -> IndividualsDataset:
    logger.info("Creating dataset")
    return IndividualsDataset(path)
