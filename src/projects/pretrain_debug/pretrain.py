import logging
from pathlib import Path

from snp_transformer.train import train

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"
import torch

torch.arange(0, 32, 2, device=torch.device("cuda"))
train(path)
