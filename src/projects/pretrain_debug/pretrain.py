import logging
from pathlib import Path
import os
from snp_transformer.train import train

logging.basicConfig(level=logging.INFO)
path = Path(__file__).parent / "config.cfg"
import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.arange(0, 32, 2, device=torch.device("cuda"))
print("created obj on cuda")
train(path)


# Test på en anden GPU!
# hmm der er måske noget med indexing af phenotype_value_ids som er -1?