import torch
from torch import nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
