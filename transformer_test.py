import torch
from tqdm import tqdm

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=512)
transformer = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3).cuda()

target = torch.rand(15_000,256).cuda()
data = torch.rand(15_000, 256, requires_grad=True).cuda()

for i in tqdm(range(100)):
    op = transformer(data)
    loss = torch.nn.functional.mse_loss(op, target)
    loss.backward()
