import torch
import pandas as pd
import numpy as np
import plotly.express as px

data_points = pd.read_csv("3_Linear_Perceptomes/data_points.csv")
data_labels = pd.read_csv("3_Linear_Perceptomes/data_labels.csv")

points = torch.tensor(data_points.to_numpy())
labels = torch.tensor(data_labels.to_numpy())

