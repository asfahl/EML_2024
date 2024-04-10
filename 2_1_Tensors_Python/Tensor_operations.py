import torch
import numpy as np

# create lists
p = [[0, 1, 2], [3, 4, 5]]
q = [[6, 7, 8], [9, 10, 11]]

# create tensors
P = torch.tensor(p)
Q = torch.tensor(q)

print("Element Add")
print(P+Q)
print("Element Mult")
print(P*Q)
print("torch Add")
print(torch.add(P,Q))
print("torch Mult")
print(torch.mul(P,Q))

# torch Matmul
print("Torch Matmul")
print(torch.matmul(P, np.transpose(Q)))
print("@ Matmul")
print(P@np.transpose(Q))

# Copying tensors
t_temp_1 = P.clone().detach()
t_temp_1[:] = 0
print("original tensor")
print(P)
print("tensor temp 1")
print(t_temp_1)

t_temp_2 = P
t_temp_2[:] = 0
print("original tensor")
print(P)
print("tensor temp 2")
print(t_temp_2)

# Clone detach creates a new independant instance of the tensor, while t_temp_2 = P creates a pointer
# that references the original tensor so changes to the copy also change the original

