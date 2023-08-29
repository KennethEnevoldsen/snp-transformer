from snp_transformer import Individual, IndividualsDataset

from .utils import test_data_folder


def test_individuals_dataset(test_data_folder):
    ind_dataset = IndividualsDataset(test_data_folder / "data")

    assert len(ind_dataset) == 3
    ind = ind_dataset[0]
    assert isinstance(ind, Individual)
