import pandas as pd
from pydantic import BaseModel, Field


class SNPs(BaseModel):
    """
    The single nucleotide polymorphisms (SNPs) of an individual.
    """

    values: list[int] = Field(..., description="List of SNP values")
    chromosomes: list[str] = Field(
        ...,
        description="Chromosome ids corresponding to the SNPs",
    )
    cm: list[float] = Field(..., description="Genetic distances in centimorgans")
    bp: list[int] = Field(..., description="Base pair coordinates")
    a1: list[str] = Field(..., description="Allele 1")
    a2: list[str] = Field(..., description="Allele 2")
    gene: list[str] = Field(..., description="gene name of the SNP")
    exome: list[str] = Field(..., description="exome name of the SNP")

    def __repr_str__(self, join_str: str) -> str:
        def string_rep(a, v) -> str:  # type: ignore  # noqa
            if isinstance(v, list):
                # return the "a=[1, 2, ...]" representation with the first 2 elements
                return f"{a}=[{v[0]}, {v[1]}, ...]"
            return f"{a}={v!r}"

        return join_str.join(
            repr(v) if a is None else string_rep(a, v) for a, v in self.__repr_args__()
        )

    def __len__(self) -> int:
        return len(self.values)


class Individual(BaseModel):
    """
    A data object representing an individual.
    """

    iid: str = Field(..., description="Individual id")
    fid: str = Field(..., description="Family id")
    father: str = Field(..., description="Father id")
    mother: str = Field(..., description="Mother id")
    sex: int = Field(
        ...,
        description="Sex code ('1' = male, '2' = female, '0' = unknown)",
    )

    phenotype: dict[str, int] = Field(
        ...,
        description="Phenotype of the individual and their value",
    )
    snps: SNPs = Field(..., description="snps of the individual")

    def to_fam(self, include_phenotype: bool = False) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame with the same format as a .fam file.
        """
        fam = pd.DataFrame(
            {
                "fid": self.fid,
                "iid": self.iid,
                "father": self.father,
                "mother": self.mother,
                "sex": self.sex,
            },
        )
        if include_phenotype is False:
            fam["phenotype"] = -9
        else:
            for phenotype, value in self.phenotype.items():
                fam[phenotype] = value

        return fam

    def to_sparse(self) -> pd.DataFrame:
        """
        Convert snps to .sparse format on the following format:

        ```
        Individual SNP Value
        1 1 1
        1 2 1
        1 3 1
        1 4 1
        1 6 1
        ```
        """
        snp_ids: list[int] = list(range(len(self.snps)))
        sparse = pd.DataFrame(
            {
                "Individual": self.iid,
                "SNP": snp_ids,
                "Value": self.snps.values,
            },
        )
        return sparse

    def to_details(self) -> pd.DataFrame:
        """
        Create a .details file on the format

        chr snp_id cm bp a1 a2 gene exome
        1 8:165998:A:G 0 115998 G A NO_GENE NO_EXON
        1 8:166003:G:A 0 116003 A G NO_GENE NO_EXON
        1 8:166007:C:T 0 116007 T C NO_GENE NO_EXON
        1 8:166009:T:C 0 116009 C T NO_GENE NO_EXON
        """

        snp_ids: list[int] = list(range(len(self.snps)))

        details = pd.DataFrame(
            {
                "chr": self.snps.chromosomes,
                "snp_id": snp_ids,
                "cm": self.snps.cm,
                "bp": self.snps.bp,
                "a1": self.snps.a1,
                "a2": self.snps.a2,
                "gene": self.snps.gene,
                "exome": self.snps.exome,
            },
        )
        return details

    def __len__(self) -> int:
        return len(self.snps)
