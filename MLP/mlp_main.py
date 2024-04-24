import torch
import torch.nn as nn
import dataloaders
import torch.optim as opt
import model
import trainer
import tester


# fashionMNIST images have dimensions (1,28,28)
# batch size 64
train, test = dataloaders.fashionMNIST(64)

# initialize simple MLP model
mlp = model.MLP()
try:
    mlp.load_state_dict(torch.load("mlp.pth"))
except:
    print( "No saved module found")
print(mlp)

# train model
epochs = 10
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t_loss = trainer.train(train, model, nn.CrossEntropyLoss(), opt.SGD(mlp.parameters(), lr=0.001))
    test_loss, test_correct = tester.test(test, model, nn.CrossEntropyLoss())
    print(f"Epoch training loss: {t_loss:>7f}")
    print(f"Epoch test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Training finished")

torch.save(mlp.state_dict, "mpl.pth")
print("saved module to mlp.pth")

