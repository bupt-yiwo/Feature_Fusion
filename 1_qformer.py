import torch
from qformer import QFormer  

qformer = QFormer()
image_feats = torch.randn(2, 256, 768)  # e.g. from ViT
query_out = qformer(image_feats)       # [2, 32, 768]
print(query_out)