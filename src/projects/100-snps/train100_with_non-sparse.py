import logging
import os
from pathlib import Path

from snp_transformer.runner import run_from_config_path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


logging.basicConfig(level=logging.INFO)


# Paths to run
path = (
    Path(__file__).parent
    / "fine_tune_no_pretrain_only_100_learned_pos_encoding_non_sparse.cfg"
)
run_from_config_path(path)
