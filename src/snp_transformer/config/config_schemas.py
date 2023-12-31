from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from pydantic import BaseModel, ConfigDict

from snp_transformer.dataset.dataset import IndividualsDataset
from snp_transformer.model.task_modules import TrainableModule


class TrainerConfigSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    accelerator: str = "auto"
    strategy: str = "auto"
    devices: Union[list[int], str, int] = "auto"
    num_nodes: int = 1
    precision: str = "32-true"
    logger: WandbLogger
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = 10  # use -1 for no limit
    min_steps: Optional[int] = None
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_predict_batches: Optional[Union[int, float]] = None
    overfit_batches: Union[int, float] = 0.0
    val_check_interval: Optional[Union[int, float]] = None
    check_val_every_n_epoch: Optional[int] = 1
    num_sanity_val_steps: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    enable_checkpointing: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    enable_model_summary: Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Optional[str] = None
    default_root_dir: Optional[Path] = None
    callbacks: Optional[list[Callback]] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TrainingConfigSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: int
    num_workers_for_dataloader: int = 8
    trainer: TrainerConfigSchema
    training_dataset: IndividualsDataset
    validation_dataset: IndividualsDataset


class ApplyConfigSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: int
    num_workers_for_dataloader: int = 8
    trainer: TrainerConfigSchema
    dataset: IndividualsDataset
    output_path: Path


class ResolvedConfigSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: TrainableModule
    logger: dict[str, WandbLogger]
    train: Optional[TrainingConfigSchema] = None
    apply: Optional[ApplyConfigSchema] = None
