import logging
from pathlib import Path

from snp_transformer.runner import run_from_config_path

# import os  # noqa
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # noqa


logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"

run_from_config_path(path)
