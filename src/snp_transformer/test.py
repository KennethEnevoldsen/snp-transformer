import torch

device = torch.device("cuda")


a = torch.tensor(
    [
        [-10.3353, -28.4371, 2.0768, -4.2789, -8.6644, -6.0815],
        [-10.3353, -28.4371, 2.0768, -4.2789, -8.6644, -6.0815],
        [-10.3353, -28.4371, 2.0768, -4.2789, -8.6644, -6.0815],
        [-10.3353, -28.4371, 2.0768, -4.2789, -8.6644, -6.0815],
    ],
).to(device)
b = torch.tensor([1, -1, 1, 4]).long().to(device)

loss = torch.nn.functional.cross_entropy(a, b, ignore_index=-1)
print(loss)
