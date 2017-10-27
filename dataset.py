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

def process(style_seqs, seq_len=SEQ_LEN):
    """
    Process data. Takes a list of styles and flattens the data, returning the necessary tags.
    """
    # Flatten into compositions list
    seqs = [s for y in style_seqs for s in y]
    style_tags = torch.LongTensor([s for s, y in enumerate(style_seqs) for x in y])
    return seqs, style_tags

def validation_split(data, split=0.1):
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

def sampler(data, seq_len=SEQ_LEN):
    """
    Generates sequences of data.
    """
    seqs, style_tags = data

    if len(seqs) == 0:
        raise 'Insufficient training data.'

    # Shuffle sequences randomly
    r = list(range(len(seqs)))
    random.shuffle(r)

    for seq_id in r:
        yield (
            augment(random_subseq(seqs[seq_id], seq_len)),
            # Need to retain the tensor object. Hence slicing is used.
            torch.LongTensor(style_tags[seq_id:seq_id+1])
        )

def batcher(sampler, batch_size=BATCH_SIZE):
    """
    Bundles samples into batches
    """
    batch = []

    for sample in sampler:
        batch.append(sample)
        
        if len(batch) == batch_size:
            # Convert batch
            yield [torch.stack(x) for x in zip(*batch)]
            batch = []

    # Yield the remaining batch!
    yield [torch.stack(x) for x in zip(*batch)]

def data_it(data, seq_len=SEQ_LEN):
    """
    Iterates through each event in all songs.
    """
    seqs, style_tags = data

    for c in range(len(seqs)):
        note_seq = seqs[c]
        style_tag = style_tags[c]

        for t in range(0, len(note_seq)):
            yield (note_seq[t], style_tag)

def random_subseq(sequence, seq_len):
    """ Randomly creates a subsequence from the sequence """
    index = random.randint(0, len(sequence) - seq_len - 2)
    return sequence[index:index + seq_len]

def augment(sequence):
    """
    Takes a sequence of events and randomly perform augmentations.
    """
    # Transpose by 4 semitones at most
    transpose = random.randint(-4, 4)

    if transpose == 0:
        return sequence

    # Perform transposition (consider only notes)
    return torch.LongTensor([evt + transpose if evt < TIME_OFFSET else evt for evt in sequence])