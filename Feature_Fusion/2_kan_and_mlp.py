###############KAN###############
import torch
from kan import KAN

model = KAN(in_features=2, out_features=1, width=[64, 64])

x = torch.rand(32, 2)
y = model(x)

print(y.shape)  

###############MLP###############
import torch.nn as nn

nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# MLP(in_dim=128, hidden_dim=256, out_dim=10)   ==   model = KAN(
                                                #     in_features=128,    
                                                #     out_features=10,    
                                                #     width=[256],          
                                                #     grid_size=20          
                                                # )
