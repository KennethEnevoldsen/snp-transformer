"""
"""


from pathlib import Path

import lightning.pytorch as pl
from confection import Config
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from snp_transformer import Registry
from snp_transformer.dataset import IndividualsDataset
from snp_transformer.task_modules import TrainableModule

default_config_path = Path(__file__).parent / "default_config.cfg"


class TrainingConfigSchema(BaseModel):
    batch_size: int


class DatasetsConfigSchema(BaseModel):
    train: IndividualsDataset
    validation: IndividualsDataset


class ResolvedConfigSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: DatasetsConfigSchema
    model: TrainableModule
    training: TrainingConfigSchema


def parse_config(config_path: Path | None) -> ResolvedConfigSchema:
    if config_path is None:
        config_path = default_config_path
    cfg = Config().from_disk(default_config_path)

    resolved = Registry.resolve(cfg)
    return ResolvedConfigSchema(**resolved)


def train(config_path: Path | None = None) -> None:
    config = parse_config(config_path)

    training_cfg = config.training

    model = config.model
    embedder = model.embedding_module

    training_dataset = config.dataset.train
    validation_dataset = config.dataset.validation

    # create dataloader:
    train_loader = DataLoader(
        training_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=embedder.collate_individuals,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=embedder.collate_individuals,
    )

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
