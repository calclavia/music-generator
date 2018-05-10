import os
import argparse
import numpy as np
import torch
from midi_io import load_midi
from model import EncoderRNN, DecoderRNN, DeepJ
from dataset import *
from constants import *

def main():
    parser = argparse.ArgumentParser(description='Apply Spherical Linear Interpolation between two pieces of music')
    parser.add_argument('--model', help='Path to existing model')
    parser.add_argument('--a', help='Path to first piece of music')
    parser.add_argument('--b', help='Path to second piece of music')
    parser.add_argument('--mu', default=0, type=float, help='Scalar representing how much to interpolate between song A and B')
    args = parser.parse_args()
    mu = args.mu

    # Load model
    model = DeepJ()
    if torch.cuda.is_available():
        model.cuda()
    if args.model:
        model.load_state_dict(torch.load(args.model))
    seq_a = []
    seq_b = []
    
    try:
        # Convert songs to lists of midi events
        seq_a = load_midi(args.a)
        seq_b = load_midi(args.b)        
    except Exception as e:
        print('Error: ', e)

    a = var(torch.from_numpy(seq_a).long().unsqueeze(0), volatile=True)
    b = var(torch.from_numpy(seq_b).long().unsqueeze(0), volatile=True)
    a = model.embd(a)
    b = model.embd(b)
    latent_a, _, _ = model.encoder(a, None)
    latent_b, _, _ = model.encoder(b, None)
    latent_a = latent_a.squeeze(0).data.cpu().numpy()
    latent_b = latent_b.squeeze(0).data.cpu().numpy()
    # Normalize latent codes
    latent_a = latent_a / len(latent_a)
    latent_b = latent_b / len(latent_b)
    
    # Apply slerp
    theta = np.arccos(np.dot(latent_a, latent_b))
    slerp = (np.sin((1 - mu) * theta) / np.sin(theta)) * latent_a + (np.sin(mu * theta) / np.sin(theta)) * latent_b
    
    # Generate slerp latent vector and write to file
    with open('out/slerp.tsv', 'w') as f:
        print('Writing slerp file to out: {}'.format('SLERP'))
        f.write('\t'.join(map(str, slerp)))

if __name__ == '__main__':
    main()
