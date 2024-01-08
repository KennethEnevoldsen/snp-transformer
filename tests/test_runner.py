from pathlib import Path

import pytest
from snp_transformer.runner import run_from_config_path

test_configs = Path("tests") / "test_configs"
encoder_for_masked_lm = test_configs / "encoder_for_masked_lm.cfg"
encoder_for_clf = test_configs / "encoder_for_clf.cfg"
encoder_for_clf_from_checkpoint = test_configs / "encoder_for_clf_from_checkpoint.cfg"
encoder_for_clf_train_and_apply = test_configs / "encoder_for_clf_train_and_apply.cfg"
encoder_for_clf_apply = test_configs / "encoder_for_clf_apply.cfg"
encoder_for_masked_lm_w_callbacks = (
    test_configs / "encoder_for_masked_lm_w_callbacks.cfg"
)


@pytest.mark.parametrize(
    "config_path",
    [
        encoder_for_masked_lm,
        encoder_for_clf,
        encoder_for_clf_from_checkpoint,
        encoder_for_clf_train_and_apply,
        encoder_for_clf_apply,
        encoder_for_masked_lm_w_callbacks,
    ],
)
def test_run_from_checkpoint_path(config_path: Path):
    """Integration test of the pl trainer"""
    run_from_config_path(config_path)
