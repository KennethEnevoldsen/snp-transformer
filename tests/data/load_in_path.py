from ast import Str
from pathlib import Path
from typing import Any

from snp_transformer.data_objects import Individual, SNPs

data_path = Path(".")

import pandas as pd

sparse = pd.read_csv(
    data_path / "data.sparse",
    sep=" ",
    dtype={"Individual": str, "SNP": int, "Value": int},
)
details = pd.read_csv(
    data_path / "data.details",
    sep=" ",
    header=None,
    names=[
        "chr",
        "snp_id",
        "cm",
        "bp",
        "a1",
        "a2",
        "gene",
        "exome",
    ],
    dtype={
        "chr": str,
        "snp_id": str,
        "cm": float,
        "bp": int,
        "a1": str,
        "a2": str,
        "gene": str,
        "exome": str,
    },
)

fam = pd.read_csv(
    data_path / "data.fam",
    sep=" ",
    header=None,
    names=[
        "fid",
        "iid",
        "father",
        "mother",
        "sex",
        "phenotype",
    ],
    dtype={
        "fid": str,
        "iid": str,
        "father": str,
        "mother": str,
        "sex": int,
        "phenotype": float,
    },
)

fam.set_index("iid", inplace=True)


iid = "1"
ind1 = sparse[sparse["Individual"] == iid]

snp_values = ind1["Value"].values
snp_indices = ind1["SNP"].values
snp_details = details.iloc[snp_indices]
ind_fam = fam.loc[iid]
ind_fam.sex
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
