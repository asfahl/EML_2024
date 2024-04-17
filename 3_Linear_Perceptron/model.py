import torch
import torch.nn as nn

class LP(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.linear = nn.Linear(3,1)

        def forward(self, x):
            linear_output = self.linear(x)
            return linear_output