import logging
from pathlib import Path

from snp_transformer.train import train

# import os  # noqa
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # noqa


logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"

train(path)

# TODO:
#   [x] get it running for 1 full epoch
#   [x] run a model training to convergence
#   fine-tune the model
#     add phenotype prediction
#       [x] add pheno to dataset (is pheno in the dataset? - yes)
#       [x] add pheno to model
#   [x] add positional encoding


# [ ] get training to work using bf16 (current device does not support bf16, but we might get it to work on a different device) or using fp16
