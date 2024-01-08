
from pathlib import Path

import pytest
from snp_transformer import apply, run_from_config
from snp_transformer.config import load_config
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.model.task_modules.encoder_for_classification import (
    EncoderForClassification,
)

from snp_transformer.config import ApplyConfigSchema

test_folder = Path(__file__).parent

def create_new_model_checkpoint_for_test():
    config = test_folder / "test_configs" / "encoder_for_clf.cfg"
    config_dict = load_config(config)
    run_from_config(config_dict)

    checkpoint_path = Path(config_dict["training"]["trainer"]["default_root_dir"])
    checkpoints = list(checkpoint_path.glob("*.ckpt"))
    checkpoints_exist = len(checkpoints) > 0
    assert checkpoints_exist, "No checkpoints were created"

    checkpoint = checkpoints[0]
    new_checkpoint_path = test_folder / "model_checkpoints" / "encoder_for_clf.ckpt"
    checkpoint.rename(new_checkpoint_path)


@pytest.mark.parametrize("model_path", [test_folder / "model_checkpoints" / "encoder_for_clf.ckpt"])
@pytest.mark.parametrize("data_path", [test_folder / "data" /"data"])
def test_apply(model_path: Path, data_path: Path, create_new_model_checkpoint: bool = False):
    """Test that the model can be applied to a unseen dataset"""

    if create_new_model_checkpoint:
        create_new_model_checkpoint_for_test()

    model =EncoderForClassification.load_from_checkpoint(model_path)

    dataset = IndividualsDataset(data_path)

    apply(model, ApplyConfigSchema())
