import numpy as np
import tensorflow as tf
import math

from constants import *
from midi_util import *

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def build_or_load(allow_load=True):
    from model import build_models
    models = build_models()
    models[0].summary()
    if allow_load:
        try:
            models[0].load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return models

def get_all_files(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith('.mid'):
                    potential_files.append(fname)
    return potential_files
