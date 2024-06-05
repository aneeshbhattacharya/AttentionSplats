import torch
import torch.nn as nn

data = torch.rand(500_000, 256, requires_grad=True).cuda()
target = torch.rand(500_000, 256).cuda()
mlp = torch.nn.Sequential(
    nn.Linear(256,256),
    nn.LeakyReLU(),
    nn.Linear(256,256)
).cuda()

optim = torch.optim.Adam(mlp.parameters(), lr=0.0001)

print("Running")
for i in range(100):
    optim.zero_grad()
    op = mlp(data)
    loss = torch.nn.functional.mse_loss(op, target)
    loss.backward()
    optim.step()