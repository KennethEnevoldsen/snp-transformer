from pathlib import Path

import pytest
from snp_transformer.runner import run_from_config_path

test_configs = Path("tests") / "test_configs"
encoder_for_masked_lm_cfg = test_configs / "encoder_for_masked_lm.cfg"
encoder_for_clf_cfg = test_configs / "encoder_for_clf.cfg"
encoder_for_clf_from_checkpoint_cfg = test_configs / "encoder_for_clf_from_checkpoint.cfg"
encoder_for_clf_train_and_apply = test_configs / "encoder_for_clf_train_and_apply.cfg"
encoder_for_clf_from_checkpoint_apply_only = test_configs / "encoder_for_clf_from_checkpoint_apply_only.cfg"


@pytest.mark.parametrize(
    "config_path",
    [
        encoder_for_masked_lm_cfg,
        encoder_for_clf_cfg,
        encoder_for_clf_from_checkpoint_cfg,
        encoder_for_clf_train_and_apply,
        encoder_for_clf_from_checkpoint_apply_only,
    ],
)
def test_run_from_checkpoint_path(config_path: Path):
    """Integration test of the pl trainer"""
    run_from_config_path(config_path)
