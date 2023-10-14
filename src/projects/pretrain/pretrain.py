import logging
from pathlib import Path

from snp_transformer.train import train

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"

train(path)


# TODO:
# [x] get it running for 1 full epoch
# [x] run a model training to convergence
# fine-tune the model
# add phenotype prediction
# add positional encoding
