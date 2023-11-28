from pathlib import Path

from snp_transformer.dataset.loaders import load_pheno, load_pheno_folder


def test_load_pheno(test_data_folder: Path):
    phenos = load_pheno(test_data_folder / "data.pheno")

    assert isinstance(phenos, dict)
    assert len(phenos.items()) == 3
    assert set(phenos.values()) == {0, 1}


def test_load_pheno_folder(test_data_folder: Path):
    phenos = load_pheno_folder(test_data_folder / "phenos")

    assert isinstance(phenos, dict)
    assert isinstance(list(phenos.values())[0], dict)  # noqa
    assert len(phenos.items()) == 2
