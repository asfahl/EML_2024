import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dist_fkts as fkt

# Adaptet from Pytorch documentation
# generate the environment
def init_process(rank, size, fn, backend='MPI'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '0'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


# 1. Aufgabe, Prozesse erstellen
if __name__ == "__main__":
    size = 5
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, fkt.run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
### Output
#/work/EML/pytorch/torch/distributed/distributed_c10d.py:1480: UserWarning: For MPI backend, world_size (5) and rank (0) are ignored since they are assigned by the MPI runtime.
#  warnings.warn(
#Rank: 0
#Size: 1
#Rank: 0
#Size: 1
#Rank: 0
#Size: 1
#Rank: 0
#Size: 1
#Rank: 0
#Size: 1
# Why has everything rank 0?

# 2. Aufgabe, geblockte Tensoren
if __name__ == "__main__":
    size = 5
    processes = []
    #mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, fkt.blocked_tensors))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# 3. Aufgabe, ungeblockte Tensoren
if __name__ == "__main__":
    size = 5
    processes = []
    #mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, fkt.unblocked_tensors))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()