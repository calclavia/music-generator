import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
from collections import deque
from tqdm import tqdm
import argparse
import midi
import os

from constants import *
from dataset import *
from midi_util import midi_encode
from model import *

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    parser.add_argument('--gen', default=False, action='store_true', help='Generate after each epoch?')
    args = parser.parse_args()

    model = build_or_load()

    if args.train:
        train(model, args.gen)
    else:
        write_file(os.path.join(SAMPLES_DIR, 'output.mid'), generate(model))

def build_or_load(allow_load=True):
    model = build_model()
    model.summary()
    if allow_load:
        try:
            model.load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return model

def train(model, gen):
    print('Loading data')
    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN)

    def epoch_cb(epoch, _):
        if epoch % 10 == 0:
            write_file(os.path.join(SAMPLES_DIR, 'epoch_{}.mid'.format(epoch)), generate(model))

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    if gen:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    print('Training')
    model.fit(train_data, train_labels, epochs=1000, callbacks=cbs)

def generate(model, style=[0.25, 0.25, 0.25, 0.25], num_bars=16, default_temp=1):
    print('Generating')
    notes_memory = deque([np.zeros((NUM_NOTES, 2)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

    results = []
    temperature = default_temp

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # The next note being built.
        next_note = np.zeros((NUM_NOTES, 2))

        # Generate each note individually
        for n in range(NUM_NOTES):
            inputs = [
                np.array([notes_memory]),
                np.array([list(notes_memory)[1:] + [next_note]]),
                np.array([beat_memory]),
                np.array([style_memory])
            ]

            pred = np.array(model.predict(inputs))
            # We only care about the last time step
            pred = pred[0, -1, :]

            # Apply temperature
            if temperature != 1:
                # Inverse sigmoid
                x = -np.log(1 / np.array(pred) - 1)
                # Apply temperature to sigmoid function
                pred = 1 / (1 + np.exp(-x / temperature))

            # Flip notes randomly
            if np.random.random() <= pred[n, 0]:
                next_note[n, 0] = 1

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

if __name__ == '__main__':
    main()
