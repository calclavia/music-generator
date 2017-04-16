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

def random_subseq(sequence, length, division_len=NOTES_PER_BAR):
    """
    Returns the range of a random subseq
    """
    # Make random starting position of sequence
    # Will never pick the last element in sequence
    start = random.randrange(0, len(sequence) - 1 - length, division_len)
    return (start, start + length)

def random_comp_subseq(compositions, length, division_len=NOTES_PER_BAR):
    """
    Returns a random music subsequence from a list of compositions
    """
    comp_index = random.randint(0, len(compositions) - 1)
    subseq_range = random_subseq(compositions[comp_index], length, division_len)
    return comp_index, subseq_range

def load_styles(styles):
    """
    Loads all music styles into a list of compositions
    """
    style_seqs = []
    for style in styles:
        # Parallel process all files into a list of music sequences
        style_seq = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(f) for f in get_all_files([style]))
        style_seqs.append(style_seq)
    return style_seqs

def extract_style(style_seqs):
    """
    Given a list of styles, where each style contains a list of compositions,
    flattens the list of styles into just a list of compositions and returns
    a list of style tags for each time step for all compositions.
    """
    style_tags = []

    for style_id, style_seq in enumerate(style_seqs):
        style_hot = one_hot(style_id, NUM_STYLES)
        style_tags += [[style_hot for x in s] for s in style_seq]
    return style_tags

def extract_beat(compositions):
    """
    Given a list of music compositions, tag each time step with beat data.
    """
    beat_tags = []

    for comp in composition:
        beat_tags.append([compute_beat(i, NOTES_PER_BAR) for t in range(len(comp))])
    return beat_tags

def sampler(style_seqs, time_steps=TIME_STEPS):
    """
    Generates training samples.
    """
    # Flatten into compositions list
    compositions = [comp for comps in style_seqs for comp in comps]
    style_tags = extract_style(style_seqs)
    beat_tags = extract_beat(compositions)

    while True:
        comp_index, r = random_comp_subseq(compositions, time_steps, 1)
        # Slice the appropriate index
        comp = compositions[comp_index][r[0]:r[1]]
        comp_style = style_tags[comp_index][r[0]:r[1]]
        comp_beat = beat_tags[comp_index][r[0]:r[1]]

        note_labels = compositions[comp_index][r[0] + 1:r[1] + 1]
        yield [comp, comp_style, comp_beat], [note_labels]

def batcher(sampler, batch_size):
    """
    Bundles samples into batches
    """
    input_batch = []
    target_batch = []

    for input_sample, target_sample in sampler:
        input_batch.append(input_sample)
        target_batch.append(target_sample)

        if len(input_batch) == batch_size:
            # Convert batch
            input_zipped = [np.array(x) for x in zip(*input_batch)]
            target_zipped = [np.array(x) for x in zip(*target_batch)]
            yield input_zipped, target_zipped

            input_batch = []
            target_batch = []

def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first note
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)
    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:i + time_steps + 1])
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
