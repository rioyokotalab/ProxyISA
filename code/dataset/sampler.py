import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from tqdm import *

class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source
        self.ys = data_source.ys
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.ys)
        self.num_classes = len(set(self.ys))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=False)
            for i in range(self.num_groups):
                ith_class_idxs = np.nonzero(np.array(self.ys) == sampled_classes[i])[0]
                sample_seleted = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                ret.extend(np.random.permutation(sample_seleted))
            num_batches -= 1
        return iter(ret) 

class PowerBalancedSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of instances per class are sampled in the minibatch,
    while p > 0, the instances per class varies.
    """
    def __init__(self, data_source, batch_size, images_per_class=3, p=0.2, enablePower=True):
        self.targets = data_source.ys
        self.batch_size = batch_size
        self.IPC = images_per_class
        self.num_classes = len(set(self.targets))
        self.enablePower = enablePower
        self.indexMap = self.buildIndices()
        self.weights = self.calcPoweredWeights(p)

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        ret = []
        num_batches = len(self) // self.batch_size
        while num_batches > 0:
            ret.extend(self.sample_batch())
            num_batches -= 1
        return iter(ret)
            
    def calcPoweredWeights(self, p):
        freqMap = []
        for i in range(self.num_classes):
            n_i = len(self.indexMap[i])
            freqMap.append(n_i ** p)
        return np.array(freqMap) / sum(freqMap)
    
    def buildIndices(self):
        indexMap = []
        for i in range(self.num_classes):
            ith_class_idxs = np.nonzero(np.array(self.targets) == i)[0]
            indexMap.append(ith_class_idxs)
        return indexMap
    
    def sample_batch(self):
        sampledIndices = []
        if self.enablePower:
            sampledClass = np.random.choice(self.num_classes, size=self.batch_size, p=self.weights)
            for i in range(self.batch_size):
                sampledIndices.append(np.random.choice(self.indexMap[sampledClass[i]]))
        else:
            num_groups = self.batch_size // self.IPC
            sampledClass = np.random.choice(self.num_classes, size=num_groups, replace=False)
            for class_idx in sampledClass:
                sampledIndices.extend(np.random.choice(self.indexMap[class_idx], size=self.IPC))
        return sampledIndices