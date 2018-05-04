import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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

def plot_graph(training_loss, validation_loss, name):
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
    train_metrics = []
    val_metrics = []

    # Epoch loop
    while True:
        # Training
        step = 1
        total_metrics = 0

        with tqdm(range(train_len)) as t:
            t.set_description('Epoch {}'.format(epoch))
            
            for _ in t:
                data = train_batcher()
                metrics = train_step(model, data, optimizer, total_step)

                total_metrics += metrics
                avg_metrics = total_metrics / step
                t.set_postfix(ce=avg_metrics[0], kl=avg_metrics[1], kl_loss=avg_metrics[2], loss=sum((avg_metrics[0], avg_metrics[2])))

                step += 1
                total_step += 1

        train_metrics.append(avg_metrics)

        # Validation
        step = 1
        total_metrics = 0

        with tqdm(range(val_len)) as t:
            t.set_description('Validation {}'.format(epoch))

            for _ in t:
                data = val_batcher()
                metrics = val_step(model, data, total_step)
                total_metrics += metrics
                avg_metrics = total_metrics / step
                t.set_postfix(ce=avg_metrics[0], kl=avg_metrics[1], kl_loss=avg_metrics[2], loss=sum((avg_metrics[0], avg_metrics[2])))

                step += 1
            
        val_metrics.append(avg_metrics)

        if plot:
            plot_graph([sum((m[0], m[2])) for m in train_metrics], [sum((m[0], m[2])) for m in val_metrics], 'loss.png')
            for i, name in enumerate(['ce_loss.png', 'kl_loss.png']):
                plot_graph(list(zip(*train_metrics))[i], list(zip(*val_metrics))[i], name)

        # Save model
        torch.save(model.state_dict(), OUT_DIR + '/model_' + str(epoch) + '.pt')

        epoch += 1

def train_step(model, data, optimizer, total_step):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()

    loss, metrics = compute_loss(model, data, total_step)
    
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
    # torch.nn.utils.clip_grad_norm_(param_copy, GRADIENT_CLIP)
    optimizer.step()

    # Copy the parameters back into the model
    copy_in_params(model, param_copy)
    return metrics

def val_step(model, data, total_step):
    model.eval()
    return compute_loss(model, data, total_step, volatile=True)[1]

def compute_loss(model, data, total_step, volatile=False):
    """
    Trains the model on a single batch of sequence.
    """
    # Convert all tensors into variables
    note_seq, styles = data

    # Feed it to the model
    if not volatile:
        note_seq = note_seq.requires_grad_()

    note_seq = note_seq.cuda()
    batch_size = note_seq.size(0)
    output, mean, logvar = model(note_seq, None)

    # Compute the loss.
    # Note that we need to convert this back into a float because it is a large summation.
    # Otherwise, it will result in 0 gradient.
    # https://github.com/timbmg/Sentence-VAE/blob/master/train.py#L68
    ce_loss = criterion(output.view(-1, NUM_ACTIONS).float(), note_seq[:, 1:].contiguous().view(-1))

    mean = mean.float()
    logvar = logvar.float()
    kl = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
    kl_weight = KL_BETA * min(total_step / KL_ANNEAL_STEPS, 1)
    # Free bits
    zero = torch.tensor(0.0)
    
    if kl.is_cuda:
        zero = zero.cuda()

    kl_loss = kl_weight * torch.max(kl - KL_TOLERANCE, zero)
    loss = ce_loss + kl_loss

    return loss, np.array([ce_loss.item(), kl.item(), kl_loss.item()])

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='Size of the batch')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    parser.add_argument('--no-fp16', default=False, action='store_true', help='Disable FP16 training')
    args = parser.parse_args()
    args.fp16 = not args.no_fp16

    print('=== Loading Model ===')
    model = DeepJ().cuda()

    if args.fp16:
        # Wrap forward method
        # fwd = model.forward
        # model.forward = lambda x, states: fwd(x.half(), states)
        model.half()

    if args.path:
        model.load_state_dict(torch.load(args.path))
        print('Restored model from checkpoint.')

    # Construct optimizer
    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
    for param in param_copy:
        param.requires_grad = True
    optimizer = optim.Adam(param_copy, lr=args.lr, eps=1e-4)
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

    # Checks if training data sounds right.
    # for i, seq in enumerate(train_batcher()[0]):
    #     save_midi('train_seq_{}'.format(i), seq.cpu().numpy())

    print('Training Sequences:', len(train_data[0]), 'Validation Sequences:', len(val_data[0]))
    print()

    print('=== Training ===')
    train(args, model, train_batcher, TRAIN_CYCLES, val_batcher, VAL_CYCLES, optimizer, plot=not args.noplot)

if __name__ == '__main__':
    main()
