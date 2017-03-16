import numpy as np
import tensorflow as tf

from music import *
from rl import A3CAgent
from midi_util import *

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def chunk(a, size):
    trim_size = (len(a) // size) * size
    return np.swapaxes(np.split(np.array(a[:trim_size]), size), 0, 1)

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
