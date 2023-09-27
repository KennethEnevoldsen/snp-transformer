import json
import logging
import shutil
from pathlib import Path

import polars as pl
from tqdm import tqdm

from snp_transformer.dataset.loaders import load_fam

logger = logging.getLogger(__name__)


def sparse_to_psparse(
    sparse_path: Path,
    psparse_path: Path,
    overwrite: bool = False,
    batch_size=50_000,
    verbose: bool = True,
) -> None:
    """
    loads in a sparse file and converts it to a psparse format. A psparse format is a
    partioned sparse format, where each individual has their own file named after their
    individual ID within a folder with the .psparse extension. The individual files are
    .sparse files.

    This function assumed that the sparse file is sorted by individual ID

    Args:
        sparse_path: path to the sparse file
        psparse_path: path to the psparse file
        overwrite: whether to overwrite the psparse file if it already exists. Defaults to False.
    """

    assert (
        sparse_path.exists()
        and sparse_path.is_file()
        and sparse_path.suffix == ".sparse"
    )

    # check that the psparse_path does not exist
    if psparse_path.exists():
        if overwrite:
            logger.warning(f"{psparse_path} already exists, overwriting...")
            shutil.rmtree(psparse_path)
        else:
            raise ValueError(
                f"{psparse_path} already exists, set overwrite=True to overwrite it",
            )

    psparse_path.with_suffix(".psparse").mkdir(exist_ok=True, parents=True)

    seen_groups = set()
    reader = pl.read_csv_batched(
        sparse_path,
        batch_size=batch_size,
        separator=" ",
    )

    batch = reader.next_batches(1)
    if verbose: 
        fam_path = psparse_path.with_suffix(".fam")
        if fam_path.exists():
            fam = load_fam(fam_path)
            total = fam.shape[0]
            pbar = tqdm(total=total, desc="Converting to psparse")
        else:
            pbar = tqdm(desc="Converting to psparse")

    while batch is not None:
        df = batch[0]
        iids = df.partition_by("Individual", as_dict=True)

        for iid, df in iids.items():
            path = psparse_path / f"{iid}.sparse"
            str_iid = str(iid)
            if str_iid in seen_groups:
                with path.open(mode="ab") as f:
                    df.write_csv(f, has_header=False, separator=" ")  # type: ignore
            else:
                df.write_csv(path, separator=" ")
                seen_groups.add(str_iid)
                if verbose:
                    pbar.update(1)  # type: ignore

        batch = reader.next_batches(1)


    # save the seen groups to a json file
    seen_groups_path = psparse_path / "individuals.json"
    with seen_groups_path.open("w") as f:
        json.dump(list(seen_groups), f)

    logger.info(f"Saved psparse file to {psparse_path}")
