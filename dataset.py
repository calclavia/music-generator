"""
Preprocesses MIDI files
"""
import numpy as np
from constants import NUM_STYLES
import math

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

def stagger(data, time_steps):
    dataX, dataY = [], []

    # First note prediction
    data = [np.zeros_like(data[0])] + list(data)

    for i in range(len(data) - time_steps - 1):
        dataX.append(data[i:(i + time_steps)])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def process(sequences, batch_size, time_steps, style):
    # Clamps the sequence
    sequences = [clamp_midi(s) for s in sequences]
    # TODO: Cirriculum training. Increasing complexity. Increasing timestep details?
    # TODO: Random transpoe?
    # TODO: Random slices of subsequence?
    train_seqs = []

    for seq in sequences:
        train_data, label_data = stagger(seq, time_steps)

        beat_data = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
        beat_data, _ = stagger(beat_data, time_steps)

        progress_data = [compute_completion(i, len(seq)) for i in range(len(seq))]
        progress_data, _ = stagger(progress_data, time_steps)

        style_data = [one_hot(style, NUM_STYLES) for i in range(len(seq))]
        style_data, _ = stagger(style_data, time_steps)

        # Chunk into batches
        train_data = chunk(train_data, batch_size)
        beat_data = chunk(beat_data, batch_size)
        progress_data = chunk(progress_data, batch_size)
        style_data = chunk(style_data, batch_size)
        label_data = chunk(label_data, batch_size)

        train_seqs.append(list(zip(train_data, beat_data, progress_data, style_data, label_data)))
    return train_seqs

def load_styles(styles):
    """
    Loads all MIDI files as a piano roll.
    """
    return [load_midi(f) for f in get_all_files(styles)]

def load_process_styles(styles, batch_size, time_steps):
    """
    Loads all MIDI files as a piano roll.
    """
    training_data = []
    for style_id, style in enumerate(styles):
        seqs = [load_midi(f) for f in get_all_files([style])]
        training_data += process(seqs, batch_size, time_steps, style_id)
    return training_data

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return np.minimum(np.ceil(sequence[:, MIN_NOTE:MAX_NOTE]), 1)

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.concatenate((np.zeros((len(sequence), MIN_NOTE)), sequence), axis=1)
