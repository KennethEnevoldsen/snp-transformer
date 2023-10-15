from pathlib import Path

import pytest
from snp_transformer.train import train


@pytest.fixture()
def config_path() -> Path:
    return Path("tests") / "test_configs" / "default_test_config.cfg"


def test_train(config_path: Path):
    """Integration test of the pl trainer"""
    train(config_path)
