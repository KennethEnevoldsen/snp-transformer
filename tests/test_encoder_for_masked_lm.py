import pytest
from snp_transformer import IndividualsDataset
from snp_transformer.model.embedders import SNPEmbedder
from snp_transformer.model.positional_embeddings import AbsolutePositionalEncoding
from snp_transformer.model.task_modules import EncoderForMaskedLM
from snp_transformer.registry import OptimizerFn
from torch import nn
from torch.utils.data import DataLoader

from .conftest import TEST_DATA_FOLDER


def dummy_training_dataset() -> IndividualsDataset:
    return IndividualsDataset(TEST_DATA_FOLDER / "data")


def long_training_dataset() -> IndividualsDataset:
    return IndividualsDataset(TEST_DATA_FOLDER / "long")


@pytest.mark.parametrize(
    "training_dataset",
    [dummy_training_dataset(), long_training_dataset()],
)
def test_model(
    training_dataset: IndividualsDataset,
    optimizer_fn: OptimizerFn,
) -> None:
    # create model:
    d_model = 32
    emb = SNPEmbedder(
        d_model=d_model,
        dropout_prob=0.1,
        max_sequence_length=128,
        positonal_embedding=AbsolutePositionalEncoding(d_model=32),
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    individuals = [training_dataset[i] for i in range(len(training_dataset))]
    emb.fit(individuals)

    mdl = EncoderForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
        create_optimizer_fn=optimizer_fn,
    )

    # create dataloader:
    dataloader = DataLoader(
        training_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=mdl.collate_fn,
    )

    # run model:
    for _ in range(10):
        for input_ids, masked_labels in dataloader:
            output = mdl(input_ids, masked_labels)
            loss = output["loss"]
            loss.backward()  # ensure that the backward pass works
