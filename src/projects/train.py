from pathlib import Path

from snp_transformer.train import train

path = Path(__file__).parent / "pretrain" / "config.cfg"

train(path)
