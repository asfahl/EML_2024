import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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

def unblocked_tensors(rank, size):
    tensor = torch.zeros((3,4))
    if rank == 0:
        tensor += 1
        dist.isend(tensor=tensor, dst=1)
    else:
        dist.irecv(tensor=tensor, src=0)
    print(f"Process with Rank {rank} has tensor {tensor} after unblocked transmission")