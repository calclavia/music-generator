"""
Preprocesses MIDI files
"""
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *
from music import MIN_NOTE, MAX_NOTE, NOTES_PER_BAR
from midi_util import load_midi
from util import chunk, get_all_files, one_hot

def compute_beat(beat, notes_in_bar):
    # TODO: Compare methods
    # Angle method
    # angle = (beat % notes_in_bar) / notes_in_bar * 2 * math.pi
    # return np.array([math.cos(angle), math.sin(angle)])
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / (len_melody - 1)])

def random_subseq(sequence, time_steps, division_len=NOTES_PER_BAR):
    # Make random starting position of sequence
    start = random.randrange(0, len(sequence) - time_steps, division_len)
    return sequence[start:start + time_steps]

def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first note
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)
    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
        # dataY.append(data[i + time_steps])
    return dataX, dataY

def load_all(styles, time_steps):
    """
    Loads all MIDI files as a piano roll.
    (For Keras)
    """
    training_data = []
    beat_data = []
    style_data = []
    training_labels = []

    for style_id, style in enumerate(styles):
        style_hot = one_hot(style_id, NUM_STYLES)
        # Parallel process all files into a list of music sequences
        seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(f) for f in get_all_files([style]))

        for seq in seqs:
            if len(seq) >= time_steps:
                # Clamp MIDI to note range
                seq = clamp_midi(seq)
                # Create training data and labels
                train_data, label_data = stagger(seq, time_steps)
                training_data += train_data
                training_labels += label_data

                beat_data += stagger([compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))], time_steps)[0]

                style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

    training_data = np.array(training_data)
    beat_data = np.array(beat_data)
    style_data = np.array(style_data)
    training_labels = np.array(training_labels)
    return [training_data, training_labels, beat_data, style_data], training_labels

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    sequence = np.minimum(np.ceil(sequence[:, MIN_NOTE:MAX_NOTE]), 1)
    assert (sequence >= 0).all()
    assert (sequence <= 1).all()
    return sequence

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.concatenate((np.zeros((len(sequence), MIN_NOTE)), sequence), axis=1)
