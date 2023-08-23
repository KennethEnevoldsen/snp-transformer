from pathlib import Path

import pytest
import torch

from snp_transformer import SNP, Individual, Individuals, SNPs


@pytest.fixture
def test_data_path():
    return Path("tests/data/")


@pytest.fixture
def test_snp_data(test_data_path) -> list[Path]:
    files_names = ["sample_chr8.details", "sample_chr1.details"]
    return [test_data_path / file_name for file_name in files_names]


@pytest.fixture
def individuals(snp_data_paths: list[Path] = test_snp_data):  # type: ignore
    inds = []
    for path in snp_data_paths:
        ind = Individuals.from_disk(path)
        inds.append(ind)
    assert len(inds) == 3  # there is three individuals in the data

    return sum(inds)


@pytest.fixture
def individual(individuals: Individuals):
    return individuals[0]


@pytest.fixture
def snps(individual: Individual):
    return individual.genotype


@pytest.fixture
def snp(snps: SNPs):
    return snps[0]


class TestIndividuals:
    def test_from_disk(
        self, snp_data_paths: list[Path] = test_snp_data  # type: ignore
    ):
        inds = []
        for path in snp_data_paths:
            ind = Individuals.from_disk(path)
            inds.append(ind)

            assert isinstance(ind, Individual)
            assert isinstance(ind.genotype, SNPs)

        assert len(inds) == 2  # there is two files in the data

    def test_individuals(self, individuals: Individuals):
        ids = set([ind.id for ind in individuals])
        assert len(ids) == len(individuals), "there is duplicated ids"

        ind = individuals[0]
        assert isinstance(ind, Individual)

    def test_individual(self, individual: Individual):
        assert isinstance(individual, Individual)
        assert isinstance(individual.genotype, SNPs)
        assert isinstance(individual.phenotype, dict)

        assert len(individual) == len(individual.genotype)


class TestSNPs:
    def test_snps(self, snps: SNPs):
        assert len(snps) == len(snps.values)

    def test_snp(self, snps: SNPs):
        snp = snps[0]
        assert isinstance(snp, SNP)
        assert isinstance(snp.bp_position, None | int)
        assert isinstance(snp.genetic_distance, None | float)
        if isinstance(snp.value, int):
            assert snp.value in [0, 1, 2]
        else:
            assert snp.value is None, "snp value must be None or and int in [0, 1, 2]"

        assert isinstance(snp.gene, None | str)
        assert isinstance(snp.exome, None | str)
        assert isinstance(snp.chromosomes, None | str)

