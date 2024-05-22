import torch
from torch import nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # flattens two dimensional image tensor into one dimensional tensor
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            # 28x28 = 784
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # Output Layer
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits
        

