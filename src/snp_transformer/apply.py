import logging

import lightning.pytorch as pl
import polars  # noqa: ICN001
from torch.utils.data import DataLoader

from snp_transformer.config.config_schemas import (
    ApplyConfigSchema,
)
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.model.task_modules.encoder_for_classification import (
    IndividualPrediction,
)
from snp_transformer.model.task_modules.trainable_modules import TrainableModule

std_logger = logging.getLogger(__name__)


def apply(model: TrainableModule, config: ApplyConfigSchema) -> None:
    assert config.trainer.devices == 1, "Apply only works with a single device"
    trainer = pl.Trainer(**config.trainer.to_dict())
    dataset: IndividualsDataset = config.dataset

    model.filter_dataset(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
        num_workers=config.num_workers_for_dataloader,
    )

    std_logger.info("Applying model to dataset")
    batch_predictions: list[list[IndividualPrediction]] = trainer.predict(model, dataloader)  # type: ignore
    predictions: list[IndividualPrediction] = [
        ind for batch in batch_predictions for ind in batch
    ]
    iids = dataset.get_iids()

    assert len(predictions) == len(iids)

    df = polars.DataFrame(predictions)
    df = df.with_columns([polars.Series("iid", iids)])

    reorder_columns = ["iid", *sorted([col for col in df.columns if col != "iid"])]
    df.select(reorder_columns).write_csv(config.output_path)

    std_logger.info(f"Saved predictions to {config.output_path}")
