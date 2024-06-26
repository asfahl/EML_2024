import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

def run(rank, size):
    print(f"Rank: {rank}")
    print(f"Size: {size}")
    return

# adapted from ollama Mistral
def initialize_distributed():
    os.environ["MASTER_ADDR"] = "localhost"  # Set the master address to localhost
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())  # Get the number of GPUs in the Grace system
    rank = int(os.environ["RANK"])  # Get the current rank from environment variable

    torch.cuda.set_device(rank)  # Set the device to the current GPU (for this rank)
    dist.init_process_group(backend="MPI")  # Initialize the MPI backend for distributed communication


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
