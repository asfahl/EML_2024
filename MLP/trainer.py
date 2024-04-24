import torch
import torch.nn as nn
import torch.optim as opt
import model

## Trains the given MLP-model.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).
#  @param io_model model which is trained.
#  @param io_optimizer.
#  @return summed loss over all training samples.
def train(i_loss_func, io_data_loader, io_model, io_optimizer):
    # switch model to training mode
    io_model.train()

    # initialize loss counter
    l_loss_total = 0

    size = len(io_data_loader.dataset)

    # iterate over batch
    for batch, (data, labels) in enumerate(io_data_loader):
        # predict label for given data 
        prediction = io_model(data)
        # calculate loss
        loss = i_loss_func(prediction, labels)
        
        # Error Backpropagation
        loss.backward()
        io_optimizer.step()
        io_optimizer.zero_grad()

        #supervise progress
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            l_loss_total += loss
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return l_loss_total