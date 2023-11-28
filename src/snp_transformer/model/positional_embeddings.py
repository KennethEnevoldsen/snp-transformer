import math
from abc import abstractmethod
from pathlib import Path
from typing import Union

import torch
from snp_transformer.registry import Registry
from torch import Tensor, nn


class PositionalEncodingModule(nn.Module):
    """
    Interface
    """

    d_model: int

    def __init__(self, *args, **kwargs):  # noqa
        super().__init__()

    @abstractmethod
    def forward(self, positions: Tensor) -> Tensor:
        ...

    def to_disk(self, path: Path):
        kwargs = self.kwargs
        model_info = {
            "class_name": type(self).__name__,
            "state_dict": self.state_dict(),
            "kwargs": kwargs,
        }
        torch.save(model_info, path.with_suffix(".pt"))

    @staticmethod
    def from_disk(path: Union[str, Path]) -> "PositionalEncodingModule":
        model_info = torch.load(path)

        class_name = model_info["class_name"]
        kwargs = model_info["kwargs"]

        # Assuming all classes are in the current namespace
        model_class = globals()[class_name]
        model_instance = model_class(**kwargs)
        model_instance.load_state_dict(model_info["state_dict"])

        return model_instance


class AbsolutePositionalEncoding(PositionalEncodingModule):
    def __init__(
        self,
        d_model: int,
        dropout_prob: float = 0.1,
        w_k_constant: float = 10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.d_model = d_model
        self.w_k_constant = w_k_constant
        self.kwargs = {
            "d_model": d_model,
            "dropout_prob": dropout_prob,
            "w_k_constant": w_k_constant,
        }

    def calculate_pe(self, positions: Tensor) -> Tensor:
        """
        Calculates the positional encoding for the given positions on
        the fly. This is more memory efficient than pre-calculating
        the positional encoding for all positions which is unfeasable
        given the maximum length.
        """
        device = positions.device
        batch_size = positions.size(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device)
            * (-math.log(self.w_k_constant) / self.d_model),
        )
        pe = torch.zeros(positions.size(0), batch_size, self.d_model, device=device)

        # make the positional broadcastable
        positions = positions.unsqueeze(-1)
        div_term = div_term.view(1, 1, -1)

        pe[:, :, 0::2] = torch.sin(positions * div_term)
        pe[:, :, 1::2] = torch.cos(positions * div_term)
        return pe

    def forward(self, positions: Tensor) -> Tensor:
        """
        Arguments:
            positions: Tensor, shape ``[batch_size, seq_len]``, where ``seq_len`` is the position ids.
        """
        batch_size, seq_len = positions.shape
        # view as batch_size x seq_len -> seq_len x batch_size
        positions = positions.view(seq_len, batch_size)
        pe = self.calculate_pe(positions)
        # view as batch_size x seq_len again
        pe = pe.view(batch_size, seq_len, -1)
        pe = self.dropout(pe)

        return pe


class tAPE(PositionalEncodingModule):
    """
    derived from:
    https://arxiv.org/pdf/2305.16642.pdf
    """

    def __init__(
        self,
        d_model: int,
        dropout_prob: float = 0.1,
        length_sequence: int = 8_482_506,
        w_k_constant: float = 10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.d_model = d_model
        self.length_sequence = length_sequence
        self.w_k_constant = w_k_constant

        self.kwargs = {
            "d_model": d_model,
            "dropout_prob": dropout_prob,
            "length_sequence": length_sequence,
            "w_k_constant": w_k_constant,
        }

    def calculate_pe(self, positions: Tensor) -> Tensor:
        """
        Calculates the positional encoding for the given positions on
        the fly. This is more memory efficient than pre-calculating
        the positional encoding for all positions which is unfeasable
        given the maximum length.
        """
        device = positions.device
        batch_size = positions.size(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device)
            * (-math.log(self.w_k_constant) / self.d_model),
        )
        pe = torch.zeros(positions.size(0), batch_size, self.d_model, device=device)
        norm_const = self.d_model / self.length_sequence

        # make the positional broadcastable
        positions = positions.unsqueeze(-1)
        div_term = div_term.view(1, 1, -1)
        pe[:, :, 0::2] = torch.sin((positions * div_term) * norm_const)
        pe[:, :, 1::2] = torch.cos((positions * div_term) * norm_const)
        return pe

    def forward(self, positions: Tensor) -> Tensor:
        """
        Arguments:
            positions: Tensor, shape ``[batch_size, seq_len]``, where ``seq_len`` is the position ids.
        """
        batch_size, seq_len = positions.shape
        # view as batch_size x seq_len -> seq_len x batch_size
        positions = positions.view(seq_len, batch_size)
        pe = self.calculate_pe(positions)
        # view as batch_size x seq_len again
        pe = pe.view(batch_size, seq_len, -1)
        pe = self.dropout(pe)
        return pe


@Registry.embedders.register("absolute_positional_embedding")
def create_absolute_positional_encoding(
    d_model: int,
    dropout_prob: float = 0.1,
    w_k_constant: float = 100_000.0,
) -> AbsolutePositionalEncoding:
    return AbsolutePositionalEncoding(
        d_model=d_model,
        dropout_prob=dropout_prob,
        w_k_constant=w_k_constant,
    )


@Registry.embedders.register("tAPE")
def create_tAPE(
    d_model: int,
    dropout_prob: float = 0.1,
    length_sequence: int = 8_482_506,
    w_k_constant: float = 100_000.0,
) -> tAPE:
    return tAPE(
        d_model=d_model,
        dropout_prob=dropout_prob,
        length_sequence=length_sequence,
        w_k_constant=w_k_constant,
    )
