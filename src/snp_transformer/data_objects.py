
from pydantic import BaseModel, Field


class SNPs(BaseModel):
    """
    The single nucleotide polymorphisms (SNPs) of an individual.
    """

    values: list[int] = Field(..., description="List of SNPs")
    chromosomes: list[str] = Field(
        ..., description="List of chromosome ids corresponding to the SNPs",
    )
    positions: list[int] = Field(
        ..., description="List of positions corresponding to the SNPs",
    )
    gene: list[str] = Field(..., description="List of genes corresponding to the SNPs")


class Individual:
    """
    A data object representing an individual.
    """

    genotype: SNPs = Field(..., description="Genotype of the individual")
    phenotype: dict[str, float] = Field(
        ..., description="Phenotype of the individual and their value",
    )
