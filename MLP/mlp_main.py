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
    print("MLP loaded successfully")
except:
    print( "No saved module found")
print(mlp)

# train model
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t_loss = trainer.train(nn.CrossEntropyLoss(), train, mlp, opt.SGD(mlp.parameters(), lr=0.001))
    test_loss, test_correct = tester.test(nn.CrossEntropyLoss(), test, mlp)
    print(f"Epoch training loss: {t_loss:>7f}")
    print(f"Epoch test Error: \n Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Training MLP finished")

torch.save(mlp.state_dict, "mpl.pth")
print("saved module to mlp.pth")

