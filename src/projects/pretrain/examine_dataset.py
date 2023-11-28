import logging
from pathlib import Path

from snp_transformer.config import load_config, parse_config

logging.basicConfig(level=logging.INFO)
config_path = Path(__file__).parent / "config.cfg"

config_dict = load_config(config_path)
config = parse_config(config_dict)

training_dataset = config.dataset.training

# How many ones vs twos?
