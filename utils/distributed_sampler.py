import math, copy
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.distributed import Sampler
import torch.distributed as dist

class DistributedSampler(Sampler):
  def __init__(self, dataset, num_replicas=None, shuffle=True, rank=None, seed=0):
    if num_replicas is None:
      if not dist.is_available():
          raise RuntimeError("Requires distributed package to be available")
      num_replicas = dist.get_world_size()
    if rank is None:
      if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
      rank = dist.get_rank()
    self.dataset = dataset
    self.num_replicas = num_replicas
    self.rank = rank
    self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
    self.total_size = self.num_samples * self.num_replicas
    self.shuffle = shuffle
    self.g = torch.Generator()
    self.g.manual_seed(seed)

  def __iter__(self):
    # deterministically shuffle based on epoch
    if self.shuffle:
      indices = torch.randperm(len(self.dataset), generator=self.g).numpy()
    else:
      indices = np.arange(len(self.dataset))

    # add extra samples to make it evenly divisible
    remain = self.total_size - len(indices)
    if remain > 0:
      indices = np.concatenate((indices, indices[:remain]))
    assert len(indices) == self.total_size

    # subsample
    if self.num_replicas > 1:
      indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    return iter(indices)

  def __len__(self):
    return self.num_samples