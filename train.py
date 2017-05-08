import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
import argparse
import midi
import os

from constants import *
from dataset import *
from generate import *
from midi_util import midi_encode
from model import *

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--gen', default=False, action='store_true', help='Generate after each epoch?')
    args = parser.parse_args()

    models = build_or_load()
    train(models, args.gen)

def train(models, gen):
    print('Loading data')
    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN)

    def epoch_cb(epoch, _):
        if epoch % 10 == 0:
            write_file(os.path.join(SAMPLES_DIR, 'epoch_{}.mid'.format(epoch)), generate(models))

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    if gen:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    print('Training')
    models[0].fit(train_data, train_labels, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
