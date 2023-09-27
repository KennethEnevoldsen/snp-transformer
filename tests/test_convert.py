import shutil
from pathlib import Path

import pytest
from snp_transformer.dataset.convert import sparse_to_psparse


@pytest.mark.parametrize("batch_size", [1, 2, 50_000])
def test_sparse_to_psparse(test_data_folder: Path, batch_size: int):
    sparse_path = test_data_folder / "data.sparse"
    psparse_path = test_data_folder / "tmp.psparse"

    if psparse_path.exists():
        shutil.rmtree(psparse_path)

    sparse_to_psparse(sparse_path, psparse_path, batch_size=batch_size)

    assert psparse_path.exists()
    assert psparse_path.is_dir()
    for i in range(1, 4):
        assert (psparse_path / f"{i}.sparse").exists()

    shutil.rmtree(psparse_path)
