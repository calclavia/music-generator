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
import itertools

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
        seq = seqs[seq_id]
        # Pick random start index
        start_index = random.randint(0, len(seq) - 1 - seq_len * 2)
        seq = seq[start_index:]
        # Apply random augmentations
        seq = augment(seq)
        # Take first N elements. After augmentation seq len changes.
        seq = itertools.islice(seq, seq_len)
        seq = gen_to_tensor(seq)
        assert seq.size() == (seq_len,), seq.size()

        return (
            seq,
            # Need to retain the tensor object. Hence slicing is used.
            torch.LongTensor(style_tags[seq_id:seq_id+1])
        )
    return sample

def batcher(sampler, batch_size, seq_len=SEQ_LEN):
    """
    Bundles samples into batches
    """
    def batch():
        batch = [sampler(seq_len) for i in range(batch_size)]
        return [torch.stack(x) for x in zip(*batch)]
    return batch 

def stretch_sequence(sequence, stretch_scale):
    """ Iterate through sequence and stretch each time shift event by a factor """
    # Accumulated time in seconds
    time_sum = 0
    seq_len = 0
    for i, evt in enumerate(sequence):
        if evt >= TIME_OFFSET and evt < VEL_OFFSET:
            # This is a time shift event
            # Convert time event to number of seconds
            # Then, accumulate the time
            time_sum += convert_time_evt_to_sec(evt)
        else:
            if i > 0:
                # Once there is a non time shift event, we take the
                # buffered time and add it with time stretch applied.
                for x in seconds_to_events(time_sum * stretch_scale):
                    yield x
                # Reset tracking variables
                time_sum = 0
            seq_len += 1
            yield evt

    # Edge case where last events are time shift events
    if time_sum > 0:
        for x in seconds_to_events(time_sum * stretch_scale):
            seq_len += 1
            yield x

    # Pad sequence with empty events if seq len not enough
    if seq_len < SEQ_LEN:
        for x in range(SEQ_LEN - seq_len):
            yield TIME_OFFSET
            
def transpose(sequence):
    """ A generator that represents the sequence. """
    # Transpose by 4 semitones at most
    transpose = random.randint(-4, 4)

    if transpose == 0:
        return sequence

    # Perform transposition (consider only notes)
    return (evt + transpose if evt < TIME_OFFSET else evt for evt in sequence)

def augment(sequence):
    """
    Takes a sequence of events and randomly perform augmentations.
    """
    sequence = transpose(sequence)
    sequence = stretch_sequence(sequence, random.uniform(1.0, 1.25))
    return sequence
