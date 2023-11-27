from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


def read_csv_with_sep_handling(
    path: Path,
    sep: Sequence[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read in a csv file with multiple possible separators.
    """
    for s in sep:
        try:
            return pd.read_csv(path, sep=s, **kwargs)
        except:  # noqa
            pass
    raise ValueError(f"Could not read file {path} with any of the separators {sep}")


def load_fam(path: Path) -> pd.DataFrame:
    """
    Load in the .fam dataset format.
    """
    fam = read_csv_with_sep_handling(
        path,
        sep=[" ", "\t"],
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
        engine="pyarrow",
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
        engine="pyarrow",
    )
    return details


def load_bim(path: Path) -> pd.DataFrame:
    """
    Load in the .bim dataset format.
    """
    bim = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "chr",
            "snp_id",
            "cm",
            "bp",
            "a1",
            "a2",
        ],
        dtype={
            "chr": str,
            "snp_id": str,
            "cm": float,
            "bp": int,
            "a1": str,
            "a2": str,
        },
    )
    return bim


def convert_bim_to_details(bim: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a .bim dataframe to a .details dataframe.
    """
    details = bim
    details["gene"] = "NO_GENE"
    details["exome"] = "NO_EXON"
    details["snp_id"] = details["snp_id"].astype(str)
    return details


def load_pheno(path: Path) -> dict[str, int]:
    """
    load a phenotype file on the form:

    ```
    1 1 1
    2 2 1
    3 3 0
    ```

    Where the columns are:
        iid fid phenotype
    """

    pheno = pl.read_csv(
        path,
        separator=" ",
        has_header=False,
        dtypes={"iid": pl.Int64, "fid": pl.Int64, "phenotype": pl.Int64},
    )
    return {  # noqa
        iid: phenotype for iid, phenotype in zip(pheno["iid"], pheno["phenotype"])
    }


def load_pheno_folder(path: Path) -> dict[str, dict[str, int]]:
    """
    Assumes a folder on the format is place as path; pheno/*.pheno files one for each phenotype, e.g. pheno/height.pheno.
    """
    assert path.is_dir(), f"Path {path} is not a directory"

    # load all .pheno
    pheno_files = list(path.glob("*.pheno"))
    pheno = {}
    for pheno_file in pheno_files:
        pheno_name = pheno_file.stem
        pheno[pheno_name] = load_pheno(pheno_file)
    return pheno


def load_sparse(path: Path) -> pl.DataFrame:
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

    Where the individual ID maps to the .fam file. I.e. the first individual in the sparse file
    is the first individual in the .fam file. So to get their iid, you can do:

        ```
        idx = 1
        fam = load_fam(fam_path)
        offset_by_one = idx - 1  # .sparse is 1-indexed
        iid = fam.iloc[offset_by_one].iid
        ```
    """
    sparse = pl.read_csv(
        path,
        separator=" ",
        dtypes={"Individual": pl.Int64, "SNP": pl.Int64, "Value": pl.Int64},
    )
    return sparse
