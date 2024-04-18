import dataloaders
import model

# fashionMNIST images have dimensions (1,28,28)
train, test = dataloaders.fashionMNIST(64)