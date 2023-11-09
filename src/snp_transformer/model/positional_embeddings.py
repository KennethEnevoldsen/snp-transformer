import math

import torch
from torch import Tensor, nn


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, w_k_const: float = 10000.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.w_k_const = w_k_const

    def calculate_pe(self, positions: Tensor) -> Tensor:
        """
        Calculates the positional encoding for the given positions on
        the fly. This is more memory efficient than pre-calculating
        the positional encoding for all positions which is unfeasable
        given the maximum length.
        """
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2)
            * (-math.log(self.w_k_const) / self.d_model),
        )
        pe = torch.zeros(positions.size(0), 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(positions * div_term)
        pe[:, 0, 1::2] = torch.cos(positions * div_term)
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


class tAPE(nn.Module):
    """
    derived from:
    https://arxiv.org/pdf/2305.16642.pdf
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        length_sequence: int = 8_482_506,
        w_k_const: float = 10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.length_sequence = length_sequence
        self.w_k_const = w_k_const

    def calculate_pe(self, positions: Tensor) -> Tensor:
        """
        Calculates the positional encoding for the given positions on
        the fly. This is more memory efficient than pre-calculating
        the positional encoding for all positions which is unfeasable
        given the maximum length.
        """
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2)
            * (-math.log(self.w_k_const) / self.d_model),
        )
        pe = torch.zeros(positions.size(0), 1, self.d_model)
        norm_const = self.d_model / self.length_sequence
        pe[:, 0, 0::2] = torch.sin((positions * div_term) * norm_const)
        pe[:, 0, 1::2] = torch.cos((positions * div_term) * norm_const)
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


if __name__ == "__main__":
    pos = AbsolutePositionalEncoding(d_model=256)
    x = torch.arange(8_482_506, 8_482_506 + 100)
    x = x.unsqueeze(0)  # batch size 1
    y = pos(x)

    import matplotlib.pyplot as plt

    arr = y.squeeze().numpy().transpose()
    plt.matshow(arr)
    plt.colorbar()
    plt.title("Positional Encoding")
    plt.show()

    # Create a plot which shows the similarity between the positional encoding
    # 2000 to all other positional encodings from 0-2000

    pos = AbsolutePositionalEncoding(d_model=1024, dropout=0.0, w_k_const=100_000.0)
    factor = 1000
    idx = 100 * factor
    start = 0
    end = 200 * factor
    y = pos(torch.arange(start, end).unsqueeze(0))

    # calculate cosine similarity
    y = y.squeeze()
    sim = torch.cosine_similarity(y, y[idx].unsqueeze(0), dim=1)
    sim = sim.squeeze().numpy()

    plt.plot(sim)

    pos1 = tAPE(d_model=512, length_sequence=1000, dropout=0.0, w_k_const=100_000.0)
    y1 = pos1(torch.arange(start, end).unsqueeze(0))

    # calculate cosine similarity
    y1 = y1.squeeze()
    sim1 = torch.cosine_similarity(y1, y1[idx].unsqueeze(0), dim=1)
    sim1 = sim1.squeeze().numpy()

    plt.plot(sim1, color="red")
