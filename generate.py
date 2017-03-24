import numpy as np
import tensorflow as tf
from collections import deque
from util import *
from midi_util import *
from music import NUM_CLASSES, NOTES_PER_BAR
from constants import *
from dataset import *
from tqdm import tqdm

class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, style, default_temp=1):
        self.notes_memory = deque([np.zeros(NUM_NOTES) for _ in range(TIME_STEPS)], maxlen=TIME_STEPS)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(TIME_STEPS)], maxlen=TIME_STEPS)
        self.style_memory = deque([style for _ in range(TIME_STEPS)], maxlen=TIME_STEPS)

        # The next note being built
        self.next_note = np.zeros(NUM_NOTES)

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(list(self.notes_memory)[1:] + [self.next_note]),
            np.array(self.beat_memory),
            np.array(self.style_memory)
        )

    def choose(self, prob, note_index):
        prob = apply_temperature(prob, self.temperature)

        # Flip on randomly
        self.next_note[note_index] = 1 if np.random.random() <= prob[note_index] else 0

    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.next_note) == 0:
            self.temperature += 0.05
        else:
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros(NUM_NOTES)
        return self.results[-1]

def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / prob - 1)
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def generate(model, default_temp=1, num_bars=8, styles=None):
    print('Generating')

    if styles is None:
        styles = all_styles()

    generations = [MusicGeneration(style) for style in styles]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = list(zip(*[g.build_inputs() for g in generations]))
            ins = [np.array(i) for i in ins]

            predictions = np.array(model.predict(ins))

            for i, g in enumerate(generations):
                # We only care about the last time step
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = SAMPLES_DIR + '/' + name + '_' + str(i) + '.mid'
        print('Writing file', fpath)
        mf = midi_encode(unclamp_midi(result))
        midi.write_midifile(fpath, mf)
