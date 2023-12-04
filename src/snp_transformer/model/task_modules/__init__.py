from .encoder_for_classification import (
    EncoderForClassification,
    create_classification_task,
    create_classification_task_from_masked_lm,
)
from .encoder_for_masked_lm import EncoderForMaskedLM, create_encoder_for_masked_lm
from .trainable_modules import Targets, TrainableModule
