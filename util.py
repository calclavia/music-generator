import numpy as np
import tensorflow as tf
import math

from constants import NUM_STYLES
from music import *
from rl import A3CAgent
from midi_util import *
import itertools

def all_styles():
    """
    Returns all combinations of style vectors for generation.
    """
    style_vectors = []

    for i in itertools.product([0, 1], repeat=NUM_STYLES):
        v = np.array(i, dtype=float)
        if np.sum(v) != 0:
            style_vectors.append(v / np.sum(v))

    return style_vectors

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def chunk(a, size):
    # Zero pad extra spaces
    target_size = math.ceil(len(a) / float(size)) * size
    inc_size = target_size - len(a)
    assert inc_size >= 0 and inc_size < size, inc_size
    a = np.array(a)
    a = np.pad(a, [(0, inc_size)] + [(0, 0) for i in range(len(a.shape) - 1)], mode='constant')
    assert a.shape[0] == target_size
    return np.swapaxes(np.split(a, size), 0, 1)

def get_all_files(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith('.mid'):
                    potential_files.append(fname)
    return potential_files

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def count_number_trainable_params():
    """
    Counts the number of trainable variables.
    """
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    """
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    """
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params
