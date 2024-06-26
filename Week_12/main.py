import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Adaptet from Pytorch documentation
def run(rank, size):
    print(f"Rank: {rank}")
    print(f"Size: {size}")
    return


def init_process(rank, size, fn, backend='MPI'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '0'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


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
### Output
#work/EML/pytorch/torch/distributed/distributed_c10d.py:1480: UserWarning: For MPI backend, world_size (2) and rank (0) are ignored since they are assigned by the MPI runtime.
#  warnings.warn(
#/work/EML/pytorch/torch/distributed/distributed_c10d.py:1480: UserWarning: For MPI backend, world_size (2) and rank (1) are ignored since they are assigned by the MPI runtime.
#  warnings.warn(
#Rank: 1
#Size: 2
#Rank: 0
#Size: 2