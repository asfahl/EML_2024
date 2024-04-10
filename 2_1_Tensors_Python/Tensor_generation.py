import torch
import pandas as pd
import numpy as np
import timeit

# generate Tensor of zeros
t1 = torch.zeros(3,3,2)
print(t1)

# generate Tensor of ones
t2 = torch.ones(3,3,2)

# generate Tensor of randome values
t3 = torch.rand(3,3,2)
print(t3)

# generate tensor of ones like existing Tensor
t4 = torch.ones_like(torch.empty(3,3,2))

# create Tensor from existing structures
arr = np.array([[3, 1, 2], [3, 4, 5], [9, 8, 7]])

start = timeit.timeit()
t5 = torch.tensor([[3, 1, 2], [3, 4, 5], [9, 8, 7]])
end = timeit.timeit()
print(f"Laufzeit für Tensor aus Python Liste:{end-start}")

start = timeit.timeit()
t6 = torch.from_numpy(arr)
end = timeit.timeit()
print(f"Laufzeit für Tensor aus NP Array:{end-start}")


# Tensor from list of lists
T = torch.empty(4,2,3)
T0 = [[0, 1, 2], [3, 4, 5]]
T1 = [[6, 7, 8], [9, 10, 11]]
T2 = [[12, 13, 14], [15, 16, 17]]
T3 = [[18, 19, 20], [21, 22, 23]]
l = [T0, T1, T2, T3]
T = torch.tensor(l)
print(T)
print(T.shape)

# Tensor from NP Array
arr = np.array(l)
T = torch.tensor(arr)
print(arr)
print(T)
print(T.shape)

