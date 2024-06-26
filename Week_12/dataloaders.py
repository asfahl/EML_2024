import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dist_fkts as fkt

class SimpleDataSet( torch.utils.data.Dataset ):
  def __init__( self, i_length ):
    self.m_length = i_length

  def __len__( self ):
    return self.m_length

  def __getitem__( self, i_idx ):
    return i_idx*10

simple_dataset = SimpleDataSet()

def init_process(rank, size, fn, backend='MPI'):
    """ Initialize the distributed environment. """
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '0'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
  

sampler = torch.utils.data.distributed.DistributedSampler(simple_dataset)
loader =  torch.utils.data.DataLoader(simple_dataset, shuffel = (sampler is None), sampler = sampler)