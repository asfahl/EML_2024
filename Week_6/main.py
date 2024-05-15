import torch
import triton
import triton.language as tl
import triton_gemm

# test implementation for triton MatMul against torch.matmul
# set dimensions L (Batchsize), M, N, K with  (M, K) x (k, N) = (N, N)
# Values see task description
L = 64
M = 196
N = 768
K = 768

# random tensor with dimensions (64, 196, 768)
A = torch.randn(L, M, K, device="cuda", dtype=torch.float32)
# random tensor with dimensions (1, 768, 768)
B = torch.randn(1, K, N, device="cuda", dtype= torch.float32)

# target tensor with dimensions (L, M, N) = (64, 196, 768)
C = torch.zeros(L, M, N, device="cuda", dtype=torch.float32)

# calculate
C_torch = torch.matmul(A, B)
C_triton = triton_gemm.matmul(A=A, B=B) # No activation implemented yet

# check shapes
print(f"Shape of C_torch: {C_torch.shape}")
print(f"Shape of C_torch: {C_torch.shape}")

# check max element-wise difference
diff = C_torch - C_triton
print(f"Max absolute difference = {max(abs(diff))}")

quantiles = [0.5, 0.25, 0.75]
ms_tri, min_ms_tri, max_ms_tri = triton.testing.do_bench(lambda: triton_gemm.matmul(A, B), quantiles=quantiles)
ms_tor, min_ms_tor, max_ms_tor = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)
print(f"Torch.matmul performance \n  ms = {ms_tor} \n min_ms = {min_ms_tor} \n max_ms = {max_ms_tor}")
print("")
print(f"triton_gemm.matmul performance \n  ms = {ms_tri} \n min_ms = {min_ms_tri} \n max_ms = {max_ms_tri}")


