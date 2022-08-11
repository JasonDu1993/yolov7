import math, copy
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.distributed import Sampler
import torch.distributed as dist


class ConcatBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, dataset_sizes, drop_last=True):
        assert len(samplers) == len(batch_sizes)
        assert len(samplers) > 0
        self.samplers = samplers
        self.batch_sizes = batch_sizes
        self.leader = samplers[0]
        self.iters = [iter(s) for s in self.samplers]
        self.cumulative_sizes = self.cumsum(dataset_sizes)
        self.drop_last = drop_last

    @staticmethod
    def cumsum(sequence):
        r, s = [0], 0
        for l in sequence:
            r.append(l + s)
            s += l
        return r

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for i, sampler in enumerate(self.samplers):
            batch_i = []
            while True:
                try:
                    idx = next(self.iters[i])
                except StopIteration:
                    assert len(batch_i) < self.batch_sizes[i]
                    self.iters[i] = iter(self.samplers[i])
                    idx = next(self.iters[i])
                    if self.drop_last:
                        batch_i = []
                batch_i.append(idx + self.cumulative_sizes[i])
                if len(batch_i) >= self.batch_sizes[i]:
                    break
            batch.extend(batch_i)
        return batch

    next = __next__  # Python 2 compatibility

    def __len__(self):
        bs = self.batch_sizes[0]
        if self.drop_last:
            return len(self.leader) // bs
        else:
            return (len(self.leader) + bs - 1) // bs
