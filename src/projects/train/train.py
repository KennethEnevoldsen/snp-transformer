import logging
from pathlib import Path

from snp_transformer.runner import run_from_config_path

import os  # noqa
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # noqa


logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "fine_tune_no_pretrain.cfg"

# run_from_config_path(path)

path = Path(__file__).parent / "fine_tune_no_pretrain_only_511.cfg"

run_from_config_path(path)

path = Path(__file__).parent / "pretrain.cfg"

# run_from_config_path(path)
