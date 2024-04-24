import torch
import torch.nn as nn
import torch.optim as opt
import model

## Tests the model
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied.
#  @param io_model model which is tested.
#  @return summed loss over all test samples, number of correctly predicted samples.
def test(i_loss_func, io_data_loader, io_model ):
    # set model to testing mode
    io_model.eval()

    # model statistics
    l_loss_total = 0
    l_n_correct = 0

    # iteration stops
    size = len(io_data_loader.dataset)
    num_batches = len(io_data_loader)

    with torch.no_grad():
        for data, labels in io_data_loader:
            prediction = io_model(data)

            # loss
            loss = i_loss_func(prediction, labels)
            l_loss_total += loss.item()

            # accuracy
            correct = (prediction.argmax(1) == labels).type(torch.float)
            l_n_correct += correct.sum().item()

    # normalize statistics
    l_loss_total /= num_batches
    l_n_correct /= size

    return l_loss_total, l_n_correct
