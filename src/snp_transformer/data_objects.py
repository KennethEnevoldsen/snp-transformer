from typing import Dict, List

from pydantic import BaseModel, Field


class SNPs(BaseModel):
    """
    The single nucleotide polymorphisms (SNPs) of an individual.
    """

    values: List[int] = Field(..., description="List of SNP values")
    chromosomes: List[str] = Field(
        ..., description="Chromosome ids corresponding to the SNPs"
    )
    cm: List[float] = Field(..., description="Genetic distances in centimorgans")
    bp: List[int] = Field(..., description="Base pair coordinates")
    a1: List[str] = Field(..., description="Allele 1")
    a2: List[str] = Field(..., description="Allele 2")
    gene: List[str] = Field(..., description="gene name of the SNP")
    exome: List[str] = Field(..., description="exome name of the SNP")

    def __repr_str__(self, join_str: str) -> str:
        def string_rep(a, v) -> str:
            if isinstance(v, list):
                # return the "a=[1, 2, ...]" representation with the first 2 elements
                return f"{a}=[{v[0]}, {v[1]}, ...]"
            return f"{a}={v!r}"

        return join_str.join(
            repr(v) if a is None else string_rep(a, v) for a, v in self.__repr_args__()
        )


class Individual(BaseModel):
    """
    A data object representing an individual.
    """

    iid: str = Field(..., description="Individual id")
    fid: str = Field(..., description="Family id")
    father: str = Field(..., description="Father id")
    mother: str = Field(..., description="Mother id")
    sex: int = Field(
        ..., description="Sex code ('1' = male, '2' = female, '0' = unknown)"
    )

    phenotype: Dict[str, float] = Field(
        ..., description="Phenotype of the individual and their value"
    )
    snps: SNPs = Field(..., description="snps of the individual")
