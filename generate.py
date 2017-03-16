import numpy as np
import tensorflow as tf
from collections import deque
from util import *
from midi_util import *
from music import NUM_CLASSES, NOTES_PER_BAR
from dataset import load_music_styles, compute_beat, build_history_buffer
from constants import *
from music import MIN_NOTE
import random
import math
import argparse
import itertools

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('model', metavar='M', type=str,
                        help='Path to the model file')
    parser.add_argument('--style', metavar='S', default=None, type=float, nargs='+',
                        help='A list that defines the weights of style')
    parser.add_argument('--bars', default=32, type=int, dest='bars',
                        help='How many bars of music to generate.')
    parser.add_argument('--prime', default=False, action='store_true',
                        help='Prime the generator with inspiration?')
    parser.add_argument('--samples', default=1, type=int, dest='samples',
                        help='Number of samples to output')
    parser.add_argument('--timesteps', metavar='t', type=int,
                        default=8,
                        help='Number of timesteps')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature used to scale randomness')

    args = parser.parse_args()

    time_steps = args.timesteps
    samples = args.samples
    bars = args.bars

    if args.style is None:
        # By default, generate all different styles
        target_styles = [np.array(i, dtype=float) for i in itertools.product([0, 1], repeat=NUM_STYLES)]
    else:
        assert len(args.style) == NUM_STYLES
        target_styles = [np.array(args.style)]


    if args.prime:
        print('Loading priming melodies')
        # Inspiration melodies
        inspirations = [x for s in load_music_styles() for x in s]

    with tf.device('/cpu:0'):
        model = load_supervised_model(time_steps, args.model)

    for i in range(samples):
        for style in target_styles:
            # Skip 0 sum style
            if np.sum(style) == 0:
                continue

            # A priming melody
            inspiration = None

            if args.prime:
                while inspiration is None or len(inspiration) < time_steps:
                    inspiration = random.choice(inspirations)

            composition = generate(model, time_steps, style / np.sum(style), bars, inspiration, args.temperature)
            #mf = midi_encode_melody(composition)
            #midi.write_midifile('out/melody {} {}.mid'.format(style.astype(int), i), mf)

            # Shift notes back up
            composition = np.concatenate((np.zeros((len(composition), MIN_NOTE)), composition), axis=1)

            mf = midi_encode(composition)
            midi.write_midifile('out/music {} {}.mid'.format(style.astype(int), i), mf)


def generate(model, time_steps, style, bars, inspiration, temperature):
    """
    Generates a sequence
    """
    print('Generating music with style {} for {} bars:'.format(style, bars))
    # Prime the time steps
    history = build_history_buffer(time_steps, NUM_CLASSES, NOTES_PER_BAR, style)

    def make_inputs():
        # return [np.repeat(np.expand_dims(x, 0), 1, axis=0) for x in zip(*history)]
        return [np.repeat(np.expand_dims(x, 0), BATCH_SIZE, axis=0) for x in zip(*history)]

    if inspiration is not None:
        print('Priming...')
        # Prime the RNN for one bar
        for i in range(NOTES_PER_BAR):
            model.predict(make_inputs())
            note_hot = inspiration[i]#one_hot(inspiration[i], NUM_CLASSES)
            beat_input = compute_beat(i, NOTES_PER_BAR)
            completion_input = np.array([i / (len(inspiration) - 1)])
            # TODO: This completion may not be good, since it resets to 0
            # later.
            history.append([note_hot, beat_input, completion_input, style])

    # Compose
    print('Composing...')
    composition = []

    N = NOTES_PER_BAR * bars
    for i in range(N):
        # Batchify the input
        results = model.predict(make_inputs())
        prob_dist = results[0]
        num_outputs = len(prob_dist)

        # Inverse sigmoid
        x = -np.log(1 / np.array(prob_dist) - 1)
        # Apply temperature to sigmoid function
        prob_dist = 1 / (1 + np.exp(-x / temperature))

        note = np.zeros(num_outputs)

        for i in range(num_outputs):
            note[i] = 1 if random.random() < prob_dist[i] else 0

        """
        note[prob_dist >= 0.5] = 1
        note[prob_dist < 0.5] = 0
        """

        beat_input = compute_beat(i, NOTES_PER_BAR)
        completion_input = np.array([i / (N - 1)])
        history.append([note, beat_input, completion_input, style])

        composition.append(note)

    model.reset_states()
    return composition

if __name__ == '__main__':
    main()
