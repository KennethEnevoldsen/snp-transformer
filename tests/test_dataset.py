from pathlib import Path

from snp_transformer import Individual, IndividualsDataset


def test_individuals_dataset(test_data_folder: Path):
    ind_dataset = IndividualsDataset(test_data_folder / "data")

    assert len(ind_dataset) == 3
    ind = ind_dataset[0]
    assert isinstance(ind, Individual)


def test_individuals_dataset_w_split(test_data_folder: Path):
    ind_dataset = IndividualsDataset(
        test_data_folder / "data",
        split_path=test_data_folder / "data_train.split",
    )

    assert len(ind_dataset) == 2
    ind = ind_dataset[0]
    assert isinstance(ind, Individual)


def test_individuals_dataset_train_test_split(test_data_folder: Path):
    ind_dataset = IndividualsDataset(test_data_folder / "data")
    ind_dataset.train_test_split(
        prob=0.8,
        train_path_prefix="tmp_train",
        test_path_prefix="tmp_test",
        seed=42,
    )

    assert len(ind_dataset) == 3, "The original dataset should not be changed"
    train_split_path = test_data_folder / "tmp_train.split"
    test_split_path = test_data_folder / "tmp_test.split"
    assert train_split_path.exists()
    assert test_split_path.exists()

    with train_split_path.open() as f:
        train_iids = f.read().split("\n")
    with test_split_path.open() as f:
        test_iids = f.read().split("\n")

    assert len(train_iids) == 2
    assert len(test_iids) == 1

    # clean up
    train_split_path.unlink()
    test_split_path.unlink()
