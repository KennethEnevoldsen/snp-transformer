import logging
from pathlib import Path

from snp_transformer.train import train

# import os  # noqa
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # noqa


logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"

train(path)

