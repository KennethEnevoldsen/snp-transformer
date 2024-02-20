import logging
from pathlib import Path

from snp_transformer.runner import run_from_config_path

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "apply_fine_tune_no_pretrain.cfg"

# run_from_config_path(path)  # noqa

path = Path(__file__).parent / "apply_fine_tune_no_pretrain_only_511.cfg"

run_from_config_path(path)
