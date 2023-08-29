from pathlib import Path

import pandas as pd
import pytest
from snp_transformer.loaders import load_details, load_fam, load_sparse


@pytest.fixture()
def test_data_folder() -> Path:
    return Path("tests/data/")


@pytest.fixture()
def fam_path(test_data_folder: Path) -> Path:
    return test_data_folder / "data.fam"


@pytest.fixture()
def sparse_path(test_data_folder: Path) -> Path:
    return test_data_folder / "data.sparse"


@pytest.fixture()
def details_path(test_data_folder: Path) -> Path:
    return test_data_folder / "data.details"


@pytest.fixture()
def details(details_path: Path) -> pd.DataFrame:
    return load_details(details_path)


@pytest.fixture()
def fam(fam_path: Path) -> pd.DataFrame:
    return load_fam(fam_path)


@pytest.fixture()
def sparse(sparse_path: Path) -> pd.DataFrame:
    return load_sparse(sparse_path)
