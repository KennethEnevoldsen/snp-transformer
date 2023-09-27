from pathlib import Path

from snp_transformer.dataset.loaders import Psparse


def test_Psparse(test_data_folder: Path):
    psparse = Psparse(test_data_folder / "data.psparse")
    assert len(psparse) == 3
    for ind in psparse.individuals:
        psparse[ind]
