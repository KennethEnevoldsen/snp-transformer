

import logging

import lightning.pytorch as pl
import polars  # noqa: ICN001
import torch
from torch.utils.data import DataLoader

from snp_transformer.config.config_schemas import (
    ApplyConfigSchema,
)
from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.model.task_modules.trainable_modules import TrainableModule

std_logger = logging.getLogger(__name__)

def apply(model: TrainableModule, config: ApplyConfigSchema) -> None:

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
    batch_predictions = trainer.predict(model, dataloader)
    predictions = torch.cat(batch_predictions, dim=0).squeeze(-1)  # type: ignore

    individuals = dataset.get_individuals()

    df = polars.DataFrame(
        {
            "iid": [ind.iid for ind in individuals],
            "pred_proba": predictions.numpy(),
        },
    )
    df.write_csv(config.output_path)
    std_logger.info(f"Saved predictions to {config.output_path}")
