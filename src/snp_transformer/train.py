"""
"""


from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from snp_transformer.config import flatten_nested_dict, load_config, parse_config


def train(config_path: Path | None = None) -> None:
    config_dict = load_config(config_path)
    config = parse_config(config_dict)

    training_cfg = config.training
    training_dataset = config.dataset.training
    validation_dataset = config.dataset.validation
    model = config.model
    logger = training_cfg.trainer.logger  # assumes only one logger
    trainer_kwargs = training_cfg.trainer.to_dict()

    # update config
    flat_config = flatten_nested_dict(config_dict)
    logger.experiment.config.update(flat_config)  # type: ignore

    # create dataloader:
    train_loader = DataLoader(
        training_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
    )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
