import os
import numpy as np
import torch
from torch.autograd import Variable
from constants import *

def find_tick_bin(ticks):
    for b, bin_ticks in enumerate(reversed(TICK_BINS)):
        if ticks >= bin_ticks:
            return len(TICK_BINS) - 1 - b
    
    raise 'Invalid ticks {}'.format(ticks)

def batch_sample(probabilities):
    """
    Samples from a batch of probabilities.
    Returns the indices chosen
    """
    batch = []

    # Iterate over batches
    for prob in probabilities:
        sampled_index = np.random.choice(len(prob), 1, p=prob)
        batch.append(sampled_index[0])
    
    return batch

def one_hot(index, size):
    x = np.zeros(size)
    x[index] = 1
    return x

def one_hot_batch(index_batch, n):
    one_hot = torch.FloatTensor(index_batch.size(0), n).zero_()
    one_hot.scatter_(1, index_batch, 1.0)
    return one_hot

def pad_before(sequence):
    """
    Pads a sequence with a zero in the first (temporal) dimension
    """
    return np.pad(sequence, ((1, 0)), 'constant')

def get_all_files(paths, ext='.mid'):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith(ext):
                    potential_files.append(fname)
    return potential_files

def to_torch(np_arr):
    return torch.from_numpy(np_arr).float()

def var(tensor, **kwargs):
    """
    Creates a Torch variable based on CUDA settings.
    """
    if torch.cuda.is_available() and not kwargs.get('use_cpu', False) and not settings['force_cpu']:
        return Variable(tensor, **kwargs).cuda()
    else:
        return Variable(tensor, **kwargs)
