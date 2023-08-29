"""
A dataset for loading in patients
"""

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from snp_transformer.data_objects import Individual, SNPs
from snp_transformer.loaders import load_details, load_fam, load_sparse


class IndividualsDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.fam_path = path.with_suffix(".fam")
        self.sparse_path = path.with_suffix(".sparse")
        self.details_path = path.with_suffix(".details")

        # ensure that they all exist
        assert self.fam_path.exists()
        assert self.sparse_path.exists()
        assert self.details_path.exists()

        self.fam = load_fam(self.fam_path)
        self.snp_details = load_details(self.details_path)
        sparse = load_sparse(self.sparse_path)
        self.idx2iid = self.fam.index.values

        # splits the sparse matrix into individual specific sections
        self.iid2sparse: dict[str, pd.DataFrame] = {
            str(iid): df for iid, df in sparse.groupby("Individual")
        }

    def __len__(self) -> int:
        return self.fam.shape[0]

    def __getitem__(self, idx: int) -> Individual:
        iid = self.idx2iid[idx]
        ind = self.iid2sparse[iid]

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
