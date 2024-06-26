import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

def run(rank, size):
    print(f"Rank: {dist.get_rank()}")
    print(f"Size: {dist.get_world_size()}")
    return

def blocked_tensors(rank, size):
    tensor = torch.zeros((3,4))
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print(f"Process with Rank {rank} has tensor {tensor} after blocked transmission")

#[login1:1032368] *** An error occurred in MPI_Isend
#[login1:1032368] *** reported by process [1848508417,0]
#[login1:1032368] *** on communicator MPI_COMM_WORLD
#[login1:1032368] *** MPI_ERR_RANK: invalid rank
#[login1:1032368] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
#[login1:1032368] ***    and potentially your MPI job)
#^CTraceback (most recent call last):
#  File "/home/qe76qox/EML_2024/Week_12/main.py", line 55, in <module>
#    p.join()

def unblocked_tensors(rank, size):
    tensor = torch.zeros((3,4))
    if rank == 0:
        tensor += 1
        dist.isend(tensor=tensor, dst=1)
    else:
        dist.irecv(tensor=tensor, src=0)
    print(f"Process with Rank {rank} has tensor {tensor} after unblocked transmission")

def summable(rank, size, tensor):
    pass
