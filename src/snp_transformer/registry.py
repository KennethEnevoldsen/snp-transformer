from typing import Any, Callable

import catalogue
import torch
from confection import registry

OptimizerFn = Callable[[Any], torch.optim.Optimizer]


class Registry(registry):
    optimizers = catalogue.create("snp_transformers", "optimizers")
    embedders = catalogue.create("snp_transformers", "embedders")

    datasets = catalogue.create("snp_transformers", "datasets")
    tasks = catalogue.create("snp_transformers", "encoders")
    layers = catalogue.create("snp_transformers", "layers")
    loggers = catalogue.create("snp_transformers", "loggers")
