import pytest
import torch
from torch import nn

from snp_transformer.model.positional_embeddings import AbsolutePositionalEncoding, tAPE


@pytest.mark.parametrize(
    "module", [AbsolutePositionalEncoding(d_model=256), tAPE(d_model=256)]
)
def test_positional_encoding(module: nn.Module):
    x = torch.arange(8_482_506, 8_482_506 + 100)
    x = x.unsqueeze(0)  # batch size 1
    y = module(x)
    assert y.shape == (1, 100, 256)
