import logging
from pathlib import Path

from snp_transformer.train import train

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "pretrain" / "config.cfg"

train(path)
