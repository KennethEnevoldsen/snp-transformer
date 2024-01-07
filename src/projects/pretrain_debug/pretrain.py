import logging
import os
from pathlib import Path

from snp_transformer.runner import run_from_config_path

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("created obj on cuda")
run_from_config_path(path)


# Test på en anden GPU!
# hmm der er måske noget med indexing af phenotype_value_ids som er -1?
