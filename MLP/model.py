import torch
from torch import nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # flattens two dimensional image tensor into one dimensional tensor
        self.flatten = nn.Flatten()
        # 28x28 = 784
        self.linear = nn.Linear(28*28, 1000)

