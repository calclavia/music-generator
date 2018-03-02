"""
Preprocesses MIDI files
"""
import math
import numpy as np
import torch

import numpy
import math
import random
from tqdm import tqdm
import multiprocessing

from constants import *
from midi_io import load_midi
from util import *

def load(styles=STYLES):
    """
    Loads all music styles into a list of compositions
    """
    style_seqs = []
    for style in styles:
        # Parallel process all files into a list of music sequences
        style_seq = []
        seq_len_sum = 0

        for f in tqdm(get_all_files([style])):
            try:
                # Pad the sequence by an empty event
                seq = load_midi(f)
                if len(seq) >= SEQ_LEN:
                    style_seq.append(torch.from_numpy(seq).long())
                    seq_len_sum += len(seq)
                else:
                    print('Ignoring {} because it is too short {}.'.format(f, len(seq)))
            except Exception as e:
                print('Unable to load {}'.format(f), e)
        
        style_seqs.append(style_seq)
        print('Loading {} MIDI file(s) with average event count {}'.format(len(style_seq), seq_len_sum / len(style_seq)))
    return style_seqs

def process(style_seqs):
    """
    Process data. Takes a list of styles and flattens the data, returning the necessary tags.
    """
    # Flatten into compositions list
    seqs = [s for y in style_seqs for s in y]
    style_tags = torch.LongTensor([s for s, y in enumerate(style_seqs) for x in y])
    return seqs, style_tags

def validation_split(data, split=0.05):
    """
    Splits the data iteration list into training and validation indices
    """
    seqs, style_tags = data

    # Shuffle sequences randomly
    r = list(range(len(seqs)))
    random.shuffle(r)

    num_val = int(math.ceil(len(r) * split))
    train_indicies = r[:-num_val]
    val_indicies = r[-num_val:]

    assert len(val_indicies) == num_val
    assert len(train_indicies) == len(r) - num_val

    train_seqs = [seqs[i] for i in train_indicies]
    val_seqs = [seqs[i] for i in val_indicies]

    train_style_tags = [style_tags[i] for i in train_indicies]
    val_style_tags = [style_tags[i] for i in val_indicies]
    
    return (train_seqs, train_style_tags), (val_seqs, val_style_tags)

def sampler(data):
    """
    Generates sequences of data.
    """
    seqs, style_tags = data

    if len(seqs) == 0:
        raise 'Insufficient training data.'

    def sample(seq_len):
        # Pick random sequence
        seq_id = random.randint(0, len(seqs) - 1)
        return (
            gen_to_tensor(augment(random_subseq(seqs[seq_id], seq_len))),
            # Need to retain the tensor object. Hence slicing is used.
            torch.LongTensor(style_tags[seq_id:seq_id+1])
        )
    return sample

def batcher(sampler):
    """
    Bundles samples into batches
    """
    def batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        batch = [sampler(seq_len) for i in range(batch_size)]
        return [torch.stack(x) for x in zip(*batch)]
    return batch 

def random_subseq(sequence, seq_len):
    """ Randomly creates a subsequence from the sequence """
    index = random.randint(0, len(sequence) - 1 - seq_len)
    return sequence[index:index + seq_len]

def stretch_sequence(sequence, stretch_scale):
    """ Iterate through sequence and stretch each time shift event by a factor """
    stretch_sequence = []
    i = 0
    while i < len(sequence):
        evt = sequence[i]
        if evt >= TIME_OFFSET and evt < VEL_OFFSET:
            stretch_time_evts = stretch_time_evt(evt, stretch_scale)
            # Add time events after time stretch to the new sequence
            for s in stretch_time_evts:
                stretch_sequence.append(s)
        else:
            stretch_sequence.append(evt)
        i += 1
    # Number of time events after time stretch may increase or decrease
    # so sequence is sliced to ensure the number of events is consistent
    return stretch_sequence[:SEQ_LEN]

def stretch_time_evt(evt, stretch_scale):
    """ Stretch time event by a constant """
    stretch_time = convert_time_evt_to_sec(evt) * stretch_scale
    standard_ticks = round(stretch_time * TICKS_PER_SEC)
    events = []
    # Add in seconds
    while standard_ticks >= 1:
        # Find the largest bin to put this time in
        tick_bin = find_tick_bin(standard_ticks)

        if tick_bin is None:
            break

        evt_index = TIME_OFFSET + tick_bin
        events.append(evt_index)
        standard_ticks -= TICK_BINS[tick_bin]

        # Approximate to the nearest tick bin instead of precise wrapping
        if standard_ticks < TICK_BINS[-1]:
            break

    return events

def augment(sequence):
    """
    Takes a sequence of events and randomly perform augmentations.
    """
    # Transpose by 4 semitones at most
    transpose = random.randint(-4, 4)

    if transpose == 0:
        return sequence

    # Perform transposition (consider only notes)
    sequence = (evt + transpose if evt < TIME_OFFSET else evt for evt in sequence)

    # Random time stretch
    stretch_multiplier = random.uniform(1.0, 1.25)

    return stretch_sequence(list(sequence), stretch_multiplier)
