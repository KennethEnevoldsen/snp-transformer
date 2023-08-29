from pathlib import Path

import pandas as pd


def load_fam(path: Path) -> pd.DataFrame:
    """
    Load in the .fam dataset format.
    """
    fam = pd.read_csv(
        path,
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
    fam = fam.set_index("iid")
    return fam


def load_details(path: Path) -> pd.DataFrame:
    """
    loading in the .details format. It is a custom format on the form:

    ```
    1 8:165998:A:G 0 115998 G A NO_GENE NO_EXON
    1 8:166003:G:A 0 116003 A G NO_GENE NO_EXON
    1 8:166007:C:T 0 116007 T C NO_GENE NO_EXON
    1 8:166009:T:C 0 116009 C T NO_GENE NO_EXON
    ```

    Where the columns are:
        chr snp_id cm bp a1 a2 gene exome
    """
    details = pd.read_csv(
        path,
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
    return details


def load_sparse(path: Path) -> pd.DataFrame:
    """
    Load in the custom .sparse format on the form:

    ```
    Individual SNP Value
    1 1 1
    1 2 1
    1 3 1
    1 4 1
    1 6 1
    2 2 1
    2 9 1
    3 3 2
    ```
    """
    sparse = pd.read_csv(
        path,
        sep=" ",
        dtype={"Individual": str, "SNP": int, "Value": int},
    )
    return sparse
