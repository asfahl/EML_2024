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