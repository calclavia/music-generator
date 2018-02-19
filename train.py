import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import argparse
import random

from dataset import *
from constants import *
from util import *
from model import DeepJ
from midi_io import save_midi

criterion = nn.CrossEntropyLoss()

def plot_loss(training_loss, validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.savefig(OUT_DIR + '/' + name)

def train(args, model, train_batcher, train_len, val_batcher, val_len, optimizer, plot=True):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    # Number of training steps per epoch
    epoch = 1
    total_step = 1

    # Keep tracks of all losses in each epoch
    train_losses = []
    val_losses = []

    # Epoch loop
    while True:
        # Training
        step = 1
        total_loss = 0

        with tqdm(range(train_len)) as t:
            t.set_description('Epoch {}'.format(epoch))
            
            for _ in t:
                data = train_batcher()
                loss = train_step(model, data, optimizer)

                total_loss += loss
                avg_loss = total_loss / step
                t.set_postfix(loss=avg_loss)

                step += 1
                total_step += 1

        train_losses.append(avg_loss)

        # Validation
        step = 1
        total_loss = 0

        with tqdm(range(val_len)) as t:
            t.set_description('Validation {}'.format(epoch))

            for _ in t:
                data = val_batcher()
                loss = val_step(model, data)
                total_loss += loss
                avg_loss = total_loss / step
                t.set_postfix(loss=avg_loss)
                step += 1
            
        val_losses.append(avg_loss)

        if plot:
            plot_loss(train_losses, val_losses, 'loss.png')

        # Save model
        torch.save(model.state_dict(), OUT_DIR + '/model_' + str(epoch) + '.pt')

        epoch += 1

def train_step(model, data, optimizer):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()

    loss, avg_loss = compute_loss(model, data)
    
    # Scale the loss
    loss = loss * SCALE_FACTOR

    # Zero out the gradient
    model.zero_grad()
    loss.backward()
    param_copy = model.param_copy
    set_grad(param_copy, list(model.parameters()))

    # Unscale the gradient
    if SCALE_FACTOR != 1:
        for param in param_copy:
            param.grad.data /= SCALE_FACTOR

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    torch.nn.utils.clip_grad_norm(model.parameters(), GRADIENT_CLIP)
    optimizer.step()

    # Copy the parameters back into the model
    copy_in_params(model, param_copy)
    return avg_loss

def val_step(model, data):
    model.eval()
    return compute_loss(model, data, volatile=True)[1]

def compute_loss(model, data, volatile=False):
    """
    Trains the model on a single batch of sequence.
    """
    # Convert all tensors into variables
    note_seq, styles = data
    styles = var(one_hot_batch(styles, NUM_STYLES), volatile=volatile)

    # Feed it to the model
    inputs = var(one_hot_seq(note_seq[:, :-1], NUM_ACTIONS), volatile=volatile)
    targets = var(note_seq[:, 1:], volatile=volatile)
    output, _ = model(inputs, styles, None)

    # Compute the loss.
    # Note that we need to convert this back into a float because it is a large summation.
    # Otherwise, it will result in 0 gradient.
    loss = criterion(output.view(-1, NUM_ACTIONS).float(), targets.contiguous().view(-1))

    return loss, loss.data[0]

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of the batch')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    parser.add_argument('--no-fp16', default=False, action='store_true', help='Disable FP16 training')
    args = parser.parse_args()
    args.fp16 = not args.no_fp16

    print('=== Loading Model ===')
    model = DeepJ()

    if torch.cuda.is_available():
        model.cuda()

        if args.fp16:
            # Wrap forward method
            fwd = model.forward
            model.forward = lambda x, style, states: fwd(x.half(), style.half(), states)
            model.half()

    if args.path:
        model.load_state_dict(torch.load(args.path))
        print('Restored model from checkpoint.')

    # Construct optimizer
    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
    for param in param_copy:
        param.requires_grad = True
    optimizer = optim.Adam(param_copy, lr=LEARNING_RATE, eps=1e-4)
    model.param_copy = param_copy

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    print('GPU: {}'.format(torch.cuda.is_available()))
    print('Batch Size: {}'.format(args.batch_size))
    print('FP16: {}'.format(args.fp16))
    print('# of Parameters: {}'.format(params))

    print()

    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    data = process(load())
    print()
    print('Creating data generators...')
    train_data, val_data = validation_split(data)
    train_batcher = batcher(sampler(train_data), args.batch_size)
    val_batcher = batcher(sampler(val_data), args.batch_size)

    """
    # Checks if training data sounds right.
    for i, (train_seq, *_) in enumerate(train_batcher()):
        save_midi('train_seq_{}'.format(i), train_seq[0].cpu().numpy())
    """

    print('Training Sequences:', len(train_data[0]), 'Validation Sequences:', len(val_data[0]))
    print()

    print('=== Training ===')
    train(args, model, train_batcher, TRAIN_CYCLES, val_batcher, VAL_CYCLES, optimizer, plot=not args.noplot)

if __name__ == '__main__':
    main()
