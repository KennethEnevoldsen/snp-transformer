from pathlib import Path

import pytest
import torch
from snp_transformer.model.embedders import Embedder, Embeddings, InputIds, SNPEmbedder
from snp_transformer.model.positional_embeddings import AbsolutePositionalEncoding


@pytest.mark.parametrize(
    ("embedding_module", "embedding_kwargs"),
    [
        (
            SNPEmbedder,
            {
                "d_model": 32,
                "dropout_prob": 0.1,
                "max_sequence_length": 128,
                "positonal_embedding": AbsolutePositionalEncoding(d_model=32),
            },
        ),
    ],
)
def test_embedding(
    individuals: list,
    embedding_module: Embedder,
    embedding_kwargs: dict,
):
    """
    Test embedding interface
    """
    batch_size = len(individuals)
    embedding_module = embedding_module(**embedding_kwargs)  # type: ignore

    embedding_module.fit(individuals)

    inputs_ids = embedding_module.collate_individuals(individuals)

    assert isinstance(inputs_ids, InputIds)
    assert isinstance(inputs_ids.snp_value_ids, torch.Tensor)
    assert isinstance(inputs_ids.is_padding, torch.Tensor)

    assert inputs_ids.snp_value_ids.shape == inputs_ids.is_padding.shape
    assert inputs_ids.snp_value_ids.shape[0] == batch_size

    max_seq_len_in_batch = inputs_ids.snp_value_ids.shape[1]

    # forward
    outputs = embedding_module(inputs_ids)

    assert isinstance(outputs, Embeddings)

    assert isinstance(outputs.embeddings, torch.Tensor)
    assert isinstance(outputs.is_padding, torch.Tensor)
    assert outputs.embeddings.shape == (
        batch_size,
        max_seq_len_in_batch,
        embedding_kwargs["d_model"],
    )


@pytest.mark.parametrize(
    ("embedding_module", "embedding_kwargs"),
    [
        (
            SNPEmbedder,
            {
                "d_model": 32,
                "dropout_prob": 0.1,
                "max_sequence_length": 128,
                "positonal_embedding": AbsolutePositionalEncoding(d_model=32),
            },
        ),
    ],
)
def test_saving_and_loading(
    embedding_module: Embedder,
    embedding_kwargs: dict,
    individuals: list,
    test_data_folder: Path,
):
    emb = SNPEmbedder(**embedding_kwargs)
    emb.fit(individuals)

    path = test_data_folder / "snp_embedder_checkpoint.pt"

    emb.to_disk(path)

    emb_loaded = embedding_module.from_disk(path)

    assert isinstance(emb_loaded, SNPEmbedder)
    assert emb_loaded.d_model == emb.d_model
    assert emb_loaded.dropout_prob == emb.dropout_prob
    assert emb_loaded.max_sequence_length == emb.max_sequence_length
    assert emb_loaded.is_fitted == emb.is_fitted
    assert emb_loaded.vocab == emb.vocab
