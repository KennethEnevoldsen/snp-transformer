from torch import nn


def create_encoder_layer(
    embedding_dim: int,
    nhead: int,
    dim_feedforward: int,
    layer_norm_eps: float = 1e-12,
    norm_first: bool = True,
):
    return nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=norm_first,
    )


def create_transformers_encoder(num_layers: int, encoder_layer: nn.Module) -> nn.Module:
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
