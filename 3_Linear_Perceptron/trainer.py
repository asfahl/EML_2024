import torch
import torch.nn as nn

## Trains the given linear perceptron.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader which provides the training data.
#  @param io_model model which is trained.
#  @param io_optimizer used optimizer.
#  @return loss.
def train(i_loss_func, io_data_loader, io_model, io_optimizer):
    # switch model to training mode
    io_model.train()

    loss_total = 0
    for batch in io_data_loader:
        io_optimizer.zero_grad()
        inputs, targets = batch
        output = io_model(inputs)
        loss = i_loss_func(output, targets)
        loss.backward()
        loss_total += loss.item()

    return loss_total