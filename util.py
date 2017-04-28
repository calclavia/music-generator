import os
import numpy as np
import torch
from torch.autograd import Variable

def one_hot(index, size):
    x = np.zeros(size)
    x[index] = 1
    return x

def pad_before(sequence):
    """
    Pads a sequence with a zero in the first (temporal) dimension
    """
    return np.pad(sequence, ((1, 0), (0, 0)), 'constant')

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
    if torch.cuda.is_available():
        return Variable(tensor, **kwargs).cuda()
    else:
        return Variable(tensor, **kwargs)
