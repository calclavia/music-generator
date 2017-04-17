import tensorflow as tf
from keras.layers import Input, Activation, LSTM, Dense, Dropout, Lambda, Reshape, Conv1D, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers.merge import Concatenate, Add, Multiply
from collections import deque
from tqdm import tqdm
import argparse

from constants import TIME_STEPS
from dataset import *
from generate import *
from music import OCTAVE, NUM_OCTAVES
from midi_util import midi_encode
from models import *
import midi

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    parser.add_argument('--gen', default=0, type=int, help='Generate per how many epochs?')
    args = parser.parse_args()

    model = build_or_load()

    if args.train:
        train(model, args.gen)
    else:
        write_file('sample', generate(model))

def build_or_load(allow_load=True):
    model = build_model()
    model.summary()
    if allow_load:
        try:
            model.load_weights('out/model.h5', by_name=True)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return model

def train(model, gen):
    print('Training')
    train_data, train_labels = load_all(styles, TIME_STEPS)

    def epoch_cb(epoch, _):
        if epoch % gen == 0:
            write_file('result_epoch_{}'.format(epoch), generate(model))

    cbs = [
        ModelCheckpoint('out/model.h5', monitor='loss', save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(patience=2),
        EarlyStopping(patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    if gen > 0:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    model.fit(train_data, train_labels, validation_split=0.1, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
