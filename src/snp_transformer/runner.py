import logging
from pathlib import Path

import lightning.pytorch as pl
from confection import Config
from torch.utils.data import DataLoader

from snp_transformer.config import (
    flatten_nested_dict,
    load_config,
    parse_config,
)
from snp_transformer.config.config_schemas import (
    ResolvedConfigSchema,
    TrainingConfigSchema,
)
from snp_transformer.model.task_modules.trainable_modules import TrainableModule

from .apply import apply

std_logger = logging.getLogger(__name__)


def run_from_config(config: Config) -> None:
    resolved_cfg: ResolvedConfigSchema = parse_config(config)

    logger = resolved_cfg.logger["logger"]

    # update config
    flat_config = flatten_nested_dict(config)

    # a hack to get the wandb logger to work
    logger._wandb_init["config"] = flat_config
    resolved_cfg.logger["logger"] = logger

    model = resolved_cfg.model

    if resolved_cfg.train:
        resolved_cfg.train.trainer.logger = logger
        _train(model, config=resolved_cfg.train)

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
    train_sampler = training_dataset.create_weighted_sampler()
    if train_sampler is None:  # noqa
        shuffle = True
    else:
        shuffle = None

    train_loader = DataLoader(
        training_dataset,
        batch_size=training_cfg.batch_size,
        collate_fn=model.collate_fn,
        shuffle=shuffle,
        num_workers=training_cfg.num_workers_for_dataloader,
        sampler=train_sampler,
    )
    val_sampler = validation_dataset.create_weighted_sampler()
    if val_sampler is None:  # noqa
        shuffle = False
    else:
        shuffle = None

    val_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=shuffle,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
        sampler=val_sampler,
    )

    trainer = pl.Trainer(**trainer_kwargs)

    std_logger.info("Starting training")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    std_logger.info("Finished training")


def run_from_config_path(config_path: Path) -> None:
    config_dict = load_config(config_path)
    run_from_config(config_dict)
