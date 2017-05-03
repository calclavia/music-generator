import numpy as np
from collections import deque
import midi

from constants import *
from dataset import *
from tqdm import tqdm
from midi_util import midi_encode

def generate(model, style=[0.25, 0.25, 0.25, 0.25], num_bars=16, default_temp=1):
    print('Generating')
    notes_memory = deque([np.zeros((NUM_NOTES, 3)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    beat_memory = deque([np.zeros_like(compute_beat(0, NOTES_PER_BAR)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

    results = []
    temperature = default_temp

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # The next note being built.
        next_note = np.zeros((NUM_NOTES, 3))

        # Generate each note individually
        for n in range(NUM_NOTES):
            inputs = [
                np.array([notes_memory]),
                np.array([list(notes_memory)[1:] + [next_note]]),
                np.array([beat_memory]),
                np.array([style_memory])
            ]

            pred = model.predict(inputs)
            pred = np.array(pred)
            # We only care about the last time step
            pred = pred[0, -1, :]
            pred_probs = pred[n, :-1]
            vol = pred[n, -1:]

            # Apply temperature
            if temperature != 1:
                # Inverse sigmoid
                x = -np.log(1 / np.array(pred_probs) - 1)
                # Apply temperature to sigmoid function
                pred_probs = 1 / (1 + np.exp(-x / temperature))

            # Flip notes randomly
            if np.random.random() <= pred[n, 0]:
                next_note[n, 0] = 1
                # Set volume
                next_note[n, 2] = vol

                if np.random.random() <= pred[n, 1]:
                    next_note[n, 1] = 1

        # Increase temperature while silent.
        if np.count_nonzero(next_note) == 0:
            temperature += 0.05
        else:
            temperature = default_temp

        notes_memory.append(next_note)
        # Consistent with dataset representation
        beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        results.append(next_note)
    return results

def write_file(name, results):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    mf = midi_encode(unclamp_midi(results))
    midi.write_midifile(name, mf)
