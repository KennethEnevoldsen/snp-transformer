"""
"""


import logging
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from confection import Config
from torch.utils.data import DataLoader

from snp_transformer.config import (
    flatten_nested_dict,
    load_config,
    parse_config,
)
from snp_transformer.config.config_schemas import (
    ApplyConfigSchema,
    ResolvedConfigSchema,
    TrainingConfigSchema,
)
from snp_transformer.model.task_modules.trainable_modules import TrainableModule

std_logger = logging.getLogger(__name__)


def apply(model: TrainableModule, config: ApplyConfigSchema) -> None:

    trainer = pl.Trainer(**config.trainer.to_dict())
    dataset = config.dataset

    model.filter_dataset(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
        num_workers=config.num_workers_for_dataloader,
    )

    output = trainer.predict(model=model, dataloaders=dataloader)
    raise NotImplementedError
    # convert to polars --> save to disk


def run_from_config(config: Config) -> None:
    resolved_cfg: ResolvedConfigSchema = parse_config(config)

    logger = resolved_cfg.logger

    # update config
    flat_config = flatten_nested_dict(config)

    # a hack to get the wandb logger to work
    logger._wandb_init["config"] = flat_config

    model = resolved_cfg.model

    if resolved_cfg.train:
        _train(model, config = resolved_cfg.train)

    if resolved_cfg.apply is not None:
        std_logger.info("Applying model")
        apply(model, resolved_cfg.apply)

def _train(model: TrainableModule, config: TrainingConfigSchema):
    training_cfg = config
    training_dataset = config.training_dataset
    validation_dataset = config.validation_dataset
    trainer_kwargs = training_cfg.trainer.to_dict()

    # filter datasets
    model.filter_dataset(training_dataset)
    model.filter_dataset(validation_dataset)

    # create dataloader:
    train_loader = DataLoader(
        training_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
    )

    trainer = pl.Trainer(**trainer_kwargs)

    std_logger.info("Starting training")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    std_logger.info("Finished training")

def run_from_config_path(config_path: Path) -> None:
    config_dict = load_config(config_path)
    run_from_config(config_dict)
