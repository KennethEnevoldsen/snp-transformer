import logging
import os
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from snp_transformer.train import train

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"

train(path)

# TODO:
#   [x] get it running for 1 full epoch
#   [x] run a model training to convergence
#   fine-tune the model
#     add phenotype prediction
#       [ ] add pheno to dataset (is pheno in the dataset? - No)
#       [x] add pheno to model
#   [x] add positional encoding


# [ ] get training to work using bf16
#
