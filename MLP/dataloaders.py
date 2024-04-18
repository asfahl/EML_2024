import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Downloads the fashionMNIST dataset
#@param batch_size batch size of the DataLoaders
#@return training_dataloader, test_dataloader filled dataloaders with the training and test data
def fashionMNIST(batch_size):
    # download training data from PyTorchs integrated FashionMNIT dataset
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    # download test data from PyTorchs integrated FashionMNIST dataset
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    # generate DataLoaders for training and Test data from datasets
    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # check dimensions of data and labels in the training dataset
    for X,y in training_dataloader:
        print(f"Shape of data:{X.shape}")
        print(f"Shape of labels:{y.shape}")
        break

    return training_dataloader, test_dataloader


