import pytest
import torch

from snp_transformer.embedders import Embedder, SNPEmbedder, InputIds


@pytest.mark.parametrize(
    "embedding_module, embedding_kwargs",
    [(SNPEmbedder, {"d_model": 32, "dropout_prob": 0.1, "max_sequence_length": 128})],
)
def test_embeddding(
    individuals: list, embedding_module: Embedder, embedding_kwargs: dict
):
    """
    Test embedding interface
    """
    batch_size = len(individuals)

    embedding_module = embedding_module(**embedding_kwargs)  # type: ignore

    embedding_module.fit(individuals)

    inputs_ids = embedding_module.collate_individuals(individuals)

    assert isinstance(inputs_ids, InputIds)
    assert isinstance(inputs_ids["snp"], torch.Tensor)
    assert isinstance(inputs_ids.is_padding, torch.Tensor)

    assert inputs_ids["snp"].shape == inputs_ids.is_padding.shape
    assert inputs_ids["snp"].shape[0] == batch_size

    max_seq_len_in_batch = inputs_ids["snp"].shape[1]

    # forward
    outputs = embedding_module(inputs_ids)

    assert isinstance(outputs["embeddings"], torch.Tensor)
    assert isinstance(inputs_ids.is_padding, torch.Tensor)
    assert outputs["embeddings"].shape == (
        batch_size,
        max_seq_len_in_batch,
        embedding_kwargs["d_model"],
    )
