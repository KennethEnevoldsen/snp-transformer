import torch
from torch import nn
from torch.utils.data import DataLoader

from snp_transformer import IndividualsDataset
from snp_transformer.embedders import SNPEmbedder
from snp_transformer.registries import OptimizerFn
from snp_transformer.task_modules import EncoderForMaskedLM

def test_model(
    training_dataset: IndividualsDataset,
    optimizer_fn: OptimizerFn,
) -> None:
    # create model:
    d_model = 32
    emb = SNPEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    individuals = [training_dataset[i] for i in range(len(training_dataset))]
    emb.fit(individuals, add_mask_token=True)

    mdl = EncoderForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
        create_optimizer_fn=optimizer_fn,
    )

    # create dataloader:
    dataloader = DataLoader(
        training_dataset, batch_size=32, shuffle=True, collate_fn=mdl.collate_fn
    )

    # run model:
    for input_ids, masked_labels in dataloader:
        output = mdl(input_ids, masked_labels)
        loss = output["Training loss"]
        loss.backward()  # ensure that the backward pass works

