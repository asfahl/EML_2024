import torch
import numpy as np
import unittest

# Aufgabe 9.1.1
# Erstelle einen Zufallstensor (Gleichverteilt) der Größe 100 vom Typ Torch.float32 
fp32 = torch.tensor(np.random.uniform(low=-1, high=1, size=100), dtype=torch.float32)
# verschieben auf gpu
fp32.cuda()
# inplace Anwendung von nn.ReLU 
relu = torch.nn.ReLU(inplace=True)
relu(fp32)
# zurückverschieben auf CPU
fp32.cpu()

print(fp32)

# Aufgabe 9.1.2
# Erstelle 2 Zufallstensoren vom Typ brainfloat16
bf1 = torch.tensor(np.random.uniform(low=-1, high=1, size=100), dtype=torch.bfloat16)
bf2 = torch.tensor(np.random.uniform(low=-1, high=1, size=100), dtype=torch.bfloat16)

# verschiebe bf2 auf die GPU
bf2.cuda()

# Binäre Operation
print(bf1 - bf2)

# Aufgabe 9.1.3
# FP16 mantisse kann max 2**11 groß sein
a = torch.tensor(2**11 + 1, dtype=torch.float16)
b = torch.tensor(2**11 + 1, dtype=torch.float32)
print(f"a = {a}, b = {b}")

# BF16 mantisse kann max 2**8 groß sein
a = torch.tensor(2**8 + 1, dtype=torch.bfloat16)
b = torch.tensor(2**8 + 1, dtype=torch.float32)
print(f"a = {a}, b = {b}")

# TF16 mantisse kann max 2**11 groß sein, passiert auf der GPU
a = torch.tensor(2**11, dtype=torch.float32)
a.cuda()
a = a*a + 1
a.cpu()
b = torch.tensor(2**12 + 1, dtype=torch.float32)
print(f"a = {a}, b = {b}")