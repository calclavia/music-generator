import os
import numpy as np
import torch
from torch.autograd import Variable
from constants import *
import sys
from collections import defaultdict

def copy_in_params(net, params):
    """ Copies the tensor data from params to net """
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)

def set_grad(params, params_with_grad):
    """ Copies the gradient from params_with_grad to params """
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)


def ngrams(tokens, n):
    ngram = []
    for token in tokens:
        if len(ngram) < n:
            ngram.append(token)
        else:
            yield ngram
            ngram.pop(0)
            ngram.append(token)
    if len(ngram) == n:
        yield ngram


def count_ngrams(tokens, n):
    counts = defaultdict(int)
    for ngram in ngrams(tokens, n):
        counts[tuple(ngram)] += 1
    return counts


def repetitiveness(tokens, max_n=5, window_size=50):
    if len(tokens) < max_n or len(tokens) < window_size:
        raise Exception('Too few tokens, change window_size or max_n')

    result = 1.0

    for n in range(1, max_n + 1):
        numerator = 0.0
        denominator = 0.0

        for window in ngrams(tokens, window_size):
            ngram_counts = count_ngrams(window, n)
            singletons = [ngram for ngram,
                          count in ngram_counts.items() if count == 1]
            numerator += len(ngram_counts) - len(singletons)
            denominator += len(ngram_counts)

        result *= numerator / denominator

    return pow(result, 1.0 / max_n)


def autocorrelate(signal, lag=1):
    """
    Gives the correlation coefficient for the signal's correlation with itself.
    Args:
    signal: The signal on which to compute the autocorrelation. Can be a list.
    lag: The offset at which to correlate the signal with itself. E.g. if lag
        is 1, will compute the correlation between the signal and itself 1 beat
        later.
    Returns:
    Correlation coefficient.
    -1 means perfect negative correlation.
    1 means perfect positive correlation.
    """
    n = len(signal)
    x = np.asarray(signal) - np.mean(signal)
    c0 = np.var(signal) + 1e-5
    return (x[lag:] * x[:n - lag]).sum() / float(n) / c0


def gen_to_tensor(generator):
    """ Converts a generator into a Torch LongTensor """
    return torch.LongTensor(list(generator))


def convert_time_evt_to_sec(evt):
    """
    Convert time shift event to seconds
    """
    tick_bin = evt - TIME_OFFSET
    seconds = TICK_BINS[tick_bin] / TICKS_PER_SEC
    return seconds


def find_tick_bin(ticks):
    """
    Returns the tick bin this belongs to, or None if the number of ticks is too little
    """
    for b, bin_ticks in enumerate(reversed(TICK_BINS)):
        if ticks >= bin_ticks:
            return len(TICK_BINS) - 1 - b

    return None


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


def one_hot_seq(index_batch, n):
    one_hot = torch.FloatTensor(index_batch.size(
        0), index_batch.size(1), n).zero_()
    one_hot.scatter_(2, index_batch.unsqueeze(2), 1.0)
    return one_hot


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
