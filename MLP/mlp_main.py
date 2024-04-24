import dataloaders
import torch.optim as opt
import model


# fashionMNIST images have dimensions (1,28,28)
# batch size 64
train, test = dataloaders.fashionMNIST(64)

# initialize simple MLP model
mlp = model.MLP()
print(mlp)

