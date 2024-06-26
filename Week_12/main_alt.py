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

# 2. Aufgabe, geblockte Tensoren
if __name__ == "__main__":
    size = 5
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, fkt.blocked_tensors))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()