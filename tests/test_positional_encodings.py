import pytest
import torch
from snp_transformer.model.positional_embeddings import AbsolutePositionalEncoding, tAPE
from torch import nn


@pytest.mark.parametrize(
    "module",
    [AbsolutePositionalEncoding(d_model=256), tAPE(d_model=256)],
)
def test_positional_encoding(module: nn.Module):
    x = torch.arange(8_482_506, 8_482_506 + 100)
    x = x.unsqueeze(0)  # batch size 1
    # stack to have a batch size of 2
    x = torch.cat([x, x], dim=0)
    y = module(x)
    assert y.shape == (2, 100, 256)

    # test to and from disk
    torch.save(module, "test.pt")
    module_from_disk = torch.load("test.pt")

    assert isinstance(module_from_disk, nn.Module)
