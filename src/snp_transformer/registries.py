from typing import Any, Callable

import catalogue
import torch

OptimizerFn = Callable[[Any], torch.optim.Optimizer]
optimizers = catalogue.create("snp_transformers", "optimizers")
