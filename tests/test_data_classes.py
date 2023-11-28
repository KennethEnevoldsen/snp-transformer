import pandas as pd
import pytest
from snp_transformer import Individual, SNPs


@pytest.fixture()
def snps(sparse: pd.DataFrame, details: pd.DataFrame) -> SNPs:
    iid = "1"
    ind1 = sparse[sparse["Individual"] == iid]

    snp_values = ind1["Value"].values
    snp_indices = ind1["SNP"].values
    snp_details = details.iloc[snp_indices]  # type: ignore
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
    return snps


def test_individuals(fam: pd.DataFrame, snps: SNPs, iid: str = "1"):
    iid = "1"
    ind_fam = fam.loc[iid]

    individual = Individual(
        snps=snps,
        iid=iid,
        fid=ind_fam.fid,
        father=ind_fam.father,
        mother=ind_fam.mother,
        sex=ind_fam.sex,
        phenotype={},
    )

    assert isinstance(individual, Individual)
    assert isinstance(individual.snps, SNPs)
    assert isinstance(individual.phenotype, dict)

    assert len(individual) == len(individual.snps)


def test_snps(
    fam: pd.DataFrame,
    sparse: pd.DataFrame,
    details: pd.DataFrame,
):
    iid = "1"
    ind1 = sparse[sparse["Individual"] == iid]

    snp_values = ind1["Value"].values
    snp_indices = ind1["SNP"].values
    snp_details = details.iloc[snp_indices]  # type: ignore
    ind_fam = fam.loc[iid]
    assert hasattr(ind_fam, "sex")
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

    # has length attributes
    assert len(snps) == len(snps.values)
