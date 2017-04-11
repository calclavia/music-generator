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
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.style_memory)
        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.style_memory)[-1:])
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
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
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

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars=32, styles=None):
    if styles is None:
        styles = all_styles()

    print('Generating with styles:', styles)

    _, time_model, note_model = models
    generations = [MusicGeneration(style) for style in styles]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        # Pick only the last time step
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            predictions = np.array(note_model.predict(ins))

            for i, g in enumerate(generations):
                # Remove the temporal dimension
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
