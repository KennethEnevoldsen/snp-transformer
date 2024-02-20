import shutil
from pathlib import Path

import polars as pl
import pytest
from snp_transformer import apply, run_from_config
from snp_transformer.config import ApplyConfigSchema, TrainerConfigSchema, load_config
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.model.task_modules.encoder_for_classification import (
    EncoderForClassification,
)
from snp_transformer.training.loggers import create_wandb_logger

test_folder = Path(__file__).parent


def create_new_model_checkpoint_for_test():
    config = test_folder / "test_configs" / "encoder_for_clf.cfg"
    config_dict = load_config(config)
    checkpoint_path = Path(config_dict["logger"]["logger"]["save_dir"])

    # empty the folder
    shutil.rmtree(checkpoint_path)

    run_from_config(config_dict)

    checkpoints = list(checkpoint_path.glob("**/*.ckpt"))
    checkpoints_exist = len(checkpoints) > 0
    assert checkpoints_exist, "No checkpoints were created"

    checkpoint = checkpoints[0]
    new_checkpoint_path = test_folder / "model_checkpoints" / "encoder_for_clf.ckpt"
    checkpoint.rename(new_checkpoint_path)


@pytest.mark.parametrize(
    "model_path",
    [test_folder / "model_checkpoints" / "encoder_for_clf.ckpt"],
)
@pytest.mark.parametrize("data_path", [test_folder / "data" / "data"])
def test_apply(
    model_path: Path,
    data_path: Path,
    tmp_path: Path,
    create_new_model_checkpoint: bool = False,
):
    """Test that the model can be applied to a unseen dataset"""

    if create_new_model_checkpoint:
        create_new_model_checkpoint_for_test()

    model = EncoderForClassification.load_from_checkpoint(model_path)

    dataset = IndividualsDataset(data_path)

    logger = create_wandb_logger(offline=True)
    trainer_cfg = TrainerConfigSchema(logger=logger, accelerator="cpu", devices=1)

    save_path = tmp_path / "pred_proba.csv"
    apply(
        model,
        ApplyConfigSchema(
            batch_size=2,
            dataset=dataset,
            output_path=save_path,
            trainer=trainer_cfg,
        ),
    )

    assert save_path.exists()

    df = pl.read_csv(save_path)

    # check that iid == 3 has null on pheno1 and not null on pheno2
    iid3 = df.filter(df["iid"] == "3")
    assert iid3["pheno1"].is_null()[0] is True
    assert iid3["pheno2"].is_null()[0] is False
