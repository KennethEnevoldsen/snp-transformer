from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
import torch
from snp_transformer import Individual, IndividualsDataset
from snp_transformer.dataset.loaders import load_details, load_fam, load_sparse
from snp_transformer.model.optimizers import create_adam

TEST_DATA_FOLDER = Path(__file__).parent / "data"


@pytest.fixture()
def individuals(test_data_folder: Path) -> list[Individual]:
    ind_dataset = IndividualsDataset(test_data_folder / "data")

    return [ind_dataset[i] for i in range(len(ind_dataset))]


@pytest.fixture()
def training_dataset(test_data_folder: Path) -> IndividualsDataset:
    return IndividualsDataset(test_data_folder / "data")


@pytest.fixture()
def test_data_folder() -> Path:
    return TEST_DATA_FOLDER


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
    return load_sparse(sparse_path).to_pandas()


@pytest.fixture()
def optimizer_fn() -> Callable[[Any], torch.optim.Optimizer]:
    return create_adam(lr=1e-3)
