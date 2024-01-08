from pathlib import Path

from snp_transformer.dataset import IndividualsDataset

data_path = Path("/data-big-projects/snp-transformer/transfer")
dataset = IndividualsDataset(data_path / "mhc")
n_ind = len(dataset)
dataset.train_test_split(
    prob=0.8,
    train_path_prefix="train",
    test_path_prefix="test",
    seed=42,
)

# Testing that the split is correct:

train = IndividualsDataset(data_path / "mhc", split_path=data_path / "train.split")
test = IndividualsDataset(data_path / "mhc", split_path=data_path / "test.split")

assert len(train) + len(test) == n_ind
assert len(train) - int(n_ind * 0.8) < 2
assert len(test) - int(n_ind * 0.2) < 2
assert set(train.valid_iids) & set(test.valid_iids) == set()
