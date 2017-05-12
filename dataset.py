"""
Preprocesses MIDI files
"""
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from constants import *
from midi_util import load_midi
from util import *

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / len_melody])

def compute_genre(genre_id):
    """ Computes a vector that represents a particular genre """
    genre_hot = np.zeros((NUM_STYLES,))
    start_index = sum(len(s) for i, s in enumerate(styles) if i < genre_id)
    styles_in_genre = len(styles[genre_id])
    genre_hot[start_index:start_index + styles_in_genre] = 1 / styles_in_genre
    return genre_hot

def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first event
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)

    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def load_all(styles, batch_size, time_steps):
    """
    Loads all MIDI files as a piano roll.
    (For Keras)
    """
    note_data = []
    beat_data = []
    style_data = []

    note_target = []

    styles = [y for x in styles for y in x]

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
                note_data += train_data
                note_target += label_data

                beats = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
                beat_data += stagger(beats, time_steps)[0]

                style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

    note_data = np.array(note_data)
    beat_data = np.array(beat_data)
    style_data = np.array(style_data)
    note_target = np.array(note_target)
    return [note_data, note_target, beat_data, style_data], [note_target]

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')
