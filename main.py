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
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    if gen:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    print('Training')
    model.fit(train_data, train_labels, epochs=1000, callbacks=cbs)

if __name__ == '__main__':
    main()
