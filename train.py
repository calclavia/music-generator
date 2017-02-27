import numpy as np
import midi
import os
import tensorflow as tf
import os.path
import random
import itertools
import argparse
from util import *
from music import NUM_CLASSES, NOTES_PER_BAR, MAX_NOTE, NO_EVENT
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from dataset import load_styles, process_data
from tqdm import tqdm
from models import *

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--type', metavar='T', default='gru_stateful', type=str,
                        help='Type of model to use')
    parser.add_argument('--file', metavar='F', type=str,
                        default='out/model.h5',
                        help='Path to the model file')
    parser.add_argument('--timesteps', metavar='t', type=int,
                        default=8,
                        help='Number of timesteps')
    args = parser.parse_args()

    time_steps = args.timesteps

    model = load_supervised_model(time_steps, args.file, globals()[args.type])

    if args.type == 'gru_stateful':
        train_stateful(model, args.file, time_steps)
    else:
        train_stateless(model, args.file, time_steps)

def train_stateless(model, model_file, time_steps):
    input_set, target_set = process_data(load_styles(), time_steps, stateful=False)

    cbs = [
        ModelCheckpoint(filepath=model_file, monitor='acc', save_best_only=True),
        #TensorBoard(log_dir='./out/supervised/summary', histogram_freq=1),
        ReduceLROnPlateau(monitor='acc', patience=5, verbose=1),
        EarlyStopping(monitor='acc', patience=10)
    ]

    model.fit(
        input_set,
        target_set,
        nb_epoch=1000,
        callbacks=cbs
    )

def train_stateful(model, model_file, time_steps):
    sequences = process_data(load_styles(), time_steps, shuffle=False)
    # Keep track of best metrics
    best_accuracy = 0
    no_improvements = 0

    lr_patience = 4
    patience = lr_patience * 2

    for epoch in itertools.count():
        print('Epoch {}:'.format(epoch))
        acc = 0
        loss = 0
        count = 0

        order = np.random.permutation(len(sequences))
        t = tqdm(order)
        for s in t:
            inputs, targets = sequences[s]
            # Long sequence training
            for x, y in tqdm(zip(inputs, targets)):
                tr_loss, tr_acc = model.train_on_batch(x, y)

                acc += tr_acc
                loss += tr_loss
                count += 1
                t.set_postfix(loss=loss/count, acc=acc/count)
            model.reset_states()
            """
            # Bar based training
            for i, (x, y) in tqdm(enumerate(zip(inputs, targets))):
                if i % NOTES_PER_BAR == 0:
                    model.reset_states()

                tr_loss, tr_acc = model.train_on_batch(x, y)

                acc += tr_acc
                loss += tr_loss
                count += 1
                t.set_postfix(loss=loss/count, acc=acc/count)
            """

        # Save model
        if acc > best_accuracy:
            best_accuracy = acc
            no_improvements = 0
            model.save(model_file)
        else:
            no_improvements += 1

        # Lower learning rate
        if no_improvements > lr_patience:
            new_lr = K.get_value(model.optimizer.lr) * 0.2
            K.set_value(model.optimizer.lr, new_lr)
            print('Lowering learning rate to {}'.format(new_lr))

        # Stop early
        if no_improvements > patience:
            break

if __name__ == '__main__':
    main()
