"""
Preprocesses input MIDI files and converts it as a Numpy file.
"""
from preprocess import midi_io, melodies_lib
import os
import numpy as np
from music import *
from tqdm import tqdm
from collections import deque
from joblib import Parallel, delayed
import math
import random
from constants import styles


def process_melody(melody):
    """
    Converts a melody data to be compatible with the neural network.
    """
    res = []
    for x in melody:
        if x >= MAX_NOTE:
            res.append(NO_EVENT)
        elif x >= 0:
            res.append(max(x - MIN_NOTE, MIN_CLASS))
        else:
            # Apply shift to NOTE_OFF and NO_EVENT
            res.append(abs(x) - 1)
    return res


def load_melody(fname):
    try:
        seq_pb = midi_io.midi_to_sequence_proto(fname)
        melody = melodies_lib.midi_file_to_melody(
            seq_pb, steps_per_quarter=NOTES_PER_BEAT)
        melody.squash(MIN_NOTE, MAX_NOTE, 0)
        return melody
    except Exception as e:
        # print(e)
        return None


def get_all_files(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith('.mid'):
                    potential_files.append(fname)
    return potential_files


def load_then_process(f):
    melody = load_melody(f)
    if melody:
        return process_melody(melody)
    else:
        return None


def load_melodies(paths, process=True, limit=None, shuffle=True):
    assert len(paths) > 0
    files = get_all_files(paths)

    if shuffle:
        random.shuffle(files)

    if limit is not None:
        print('Limiting to {} files'.format(limit))
        files = files[:limit]

    print('Loading melodies from {} files'.format(len(files)))
    fn = load_then_process if process else load_melody
    res = Parallel(n_jobs=8, verbose=5, backend='threading')(
        delayed(fn)(f) for f in files)

    out = []
    skipped = 0
    for melody in res:
        if melody == None or len(melody) <= 1:
            skipped += 1
        else:
            out.append(melody)

    print('Loaded {} melodies (skipped {})'.format(len(out), skipped))
    return out


def compute_beat(beat, notes_in_bar):
    # TODO: Compare methods
    # Angle method
    angle = (beat % notes_in_bar) / notes_in_bar * 2 * math.pi
    return np.array([math.cos(angle), math.sin(angle)])
    # return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / (len_melody - 1)])

def build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot, prime_beats=True):
    history = deque(maxlen=time_steps)
    current_beat = NOTES_PER_BAR - 1
    for i in range(time_steps):
        assert current_beat >= 0
        beat_input = compute_beat(current_beat, notes_in_bar)
        history.appendleft([
            np.zeros(num_classes),
            beat_input if prime_beats else np.zeros_like(beat_input),
            np.zeros(1),
            style_hot
        ])

        current_beat -= 1
        if current_beat < 0:
            current_beat = NOTES_PER_BAR - 1
    return history

def melody_data_gen(melody,
                    style_hot,
                    time_steps,
                    num_classes,
                    notes_in_bar,
                    target_all):
    """
    Generates training and label data for a given melody sequence.

    Return:
        A list of samples. Each sample consist of inputs for each time step
        and a target level as a tuple.
    """
    # Recurrent history
    history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

    if target_all:
        target_history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

    for beat, note in enumerate(melody[:-1]):
        note_hot = one_hot(note, num_classes)
        beat_input = compute_beat(beat, notes_in_bar)
        completion_input = compute_completion(beat, len(melody))

        # Wrap around
        next_note_hot = one_hot(melody[beat + 1 ], num_classes)

        history.append(
            [note_hot, beat_input, completion_input, style_hot]
        )

        # Yield the current input with target
        if target_all:
            # Yield a list of targets for each time step
            target_history.append([next_note_hot])
            yield zip(*history), zip(*target_history)
        else:
            # Yield the target to predict
            yield zip(*history), [next_note_hot]


def stateless_gen(melody_styles,
                        time_steps,
                        num_classes,
                        notes_in_bar,
                        target_all=False):
    for s, style in enumerate(melody_styles):
        style_hot = one_hot(s, len(melody_styles))

        for melody in style:
            for x in melody_data_gen(melody, style_hot, time_steps, num_classes, notes_in_bar, target_all):
                yield x

def stateful_gen(melody_styles, time_steps, batch_size=1, num_classes=NUM_CLASSES, notes_in_bar=NOTES_PER_BAR, target_all=False):
    """
    For every single melody style, yield the melody along
    with its contextual inputs.
    """
    # Process the data into a list of sequences.
    # Each sequence contains input tracks
    # Each training sequence is a tuple of various inputs, including contexts
    for s, style in enumerate(melody_styles):
        style_hot = one_hot(s, len(melody_styles))

        for melody in style:
            m_data = list(melody_data_gen(melody, style_hot, time_steps, num_classes, notes_in_bar, target_all))
            # A list of sample inputs and targets
            inputs, targets = zip(*m_data)

            # Chunk input and targets into batch size
            inputs = [np.split(x, batch_size) for x in inputs]
            targets = np.split(targets, batch_size)

            yield inputs, targets

def load_styles():
    # A list of styles, each containing melodies
    return [load_melodies([style]) for style in styles]

def process_data(melody_styles, time_steps, stateful=True, limit=None, shuffle=True):
    print('Processing dataset')
    if stateful:
        return list(stateful_gen(melody_styles, time_steps))
    else:
        input_set, target_set = zip(*list(stateless_gen(melody_styles, time_steps, NUM_CLASSES, NOTES_PER_BAR)))
        input_set = [np.array(i) for i in zip(*input_set)]
        target_set = [np.array(i) for i in zip(*target_set)]
        return input_set, target_set
