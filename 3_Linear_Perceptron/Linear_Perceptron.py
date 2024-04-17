import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
import torch.nn as nn
import model
import trainer


# Load datapoints and labels as Pandas Dataframes
# Correct path for other users
data_points = pd.read_csv("3_Linear_Perceptron/data_points.csv")
data_labels = pd.read_csv("3_Linear_Perceptron/data_labels.csv")

# Generate tensors from dataframes
points = torch.tensor(data_points.to_numpy())
labels = torch.tensor(data_labels.to_numpy())

#visualize the dataset
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(points[labels[:,0]==0, 0], points[labels[:,0]==0, 2], points[labels[:,0]==0, 1], c="b")
ax.scatter(points[labels[:,0]==1, 0], points[labels[:,0]==1, 2], points[labels[:,0]==1, 1], c="g")
plt.show()
fig.savefig("Data.png")

# Wrapp data to dataset
dataset = torch.utils.data.TensorDataset(points, labels)
# turn dataset into DataLoader
training_data = torch.utils.data.DataLoader(dataset=dataset)

