from pathlib import Path

import pytest
from snp_transformer.runner import run_from_config_path

encoder_for_masked_lm_cfg = Path("tests") / "test_configs" / "encoder_for_masked_lm.cfg"
encoder_for_clf_cfg = Path("tests") / "test_configs" / "encoder_for_clf.cfg"
encoder_for_clf_from_checkpoint_cfg = (
    Path("tests") / "test_configs" / "encoder_for_clf_from_checkpoint.cfg"
)


@pytest.mark.parametrize(
    "config_path",
    [
        encoder_for_masked_lm_cfg,
        encoder_for_clf_cfg,
        encoder_for_clf_from_checkpoint_cfg,
    ],
)
def test_train(config_path: Path):
    """Integration test of the pl trainer"""
    run_from_config_path(config_path)
