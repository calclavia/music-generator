import numpy as np
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from midi_io import *
from dataset import *
from constants import *
from util import *
from model import DeepJ

class MusicGenerator():
    """
    Represents a music generation sequence
    """
    def __init__(self, model):
        self.model = model

    def generate(self, seq_len, encode_seq=None, temperature=0, latent=None, show_progress=True):
        self.model.eval()
        r = trange(seq_len) if show_progress else range(seq_len)

        seq = []
        
        if encode_seq is not None:
            # Use latent vector produced by encoder
            x = encode_seq.unsqueeze(0)
            x = self.model.embd(x)
            z, _, _ = self.model.encoder(x, None)
        else:
            # Sample latent vector
            z = torch.randn(1, self.model.latent_size)

        if latent is not None:
            # Use provided custom latent vector
            z = latent.unsqueeze(0)
        memory = None
        # Generate starting first token. Input for decoder.
        x = torch.LongTensor([[0]])
        
        for _ in r:
            logits, memory = self.model.decoder(self.model.embd(x), latent=z, hidden=memory)
            # Remove latent vector
            z = None
            # Append chosen event to sequence
            if temperature == 0:
                probs = F.softmax(logits, dim=2)
                x = torch.max(probs, 2)[1]
            else:
                probs = F.softmax(logits / temperature, dim=2)
                x = probs.squeeze(0).multinomial(1)
            seq.append(x.squeeze(0).data.numpy()[0])

        return np.array(seq)

    def export(self, name='output', seq_len=1000, show_progress=True):
        """
        Export into a MIDI file.
        """
        seq = self.generate(seq_len, show_progress=show_progress)


def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('model', help='Path to model file')
    parser.add_argument('--fname', default='sample', help='Name of the output file')
    parser.add_argument('--length', default=SEQ_LEN, type=int, help='Length of generation')
    parser.add_argument('--synth', default=False, action='store_true', help='Synthesize output in MP3')
    parser.add_argument('--encode', default=None, type=str, help='Forces the latent vector to encode a sequence')
    parser.add_argument('--temperature', default=0, type=float, help='Temperature of generation. 0 temperature = deterministic')
    parser.add_argument('--latent', default=None, help='Path to custom latent vector file')
    args = parser.parse_args()
    
    print('=== Loading Model ===')
    print('Path: {}'.format(args.model))
    settings['force_cpu'] = True
    
    model = DeepJ()

    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        print('WARNING: No model loaded! Please specify model path.')
    
    encode_seq = None
    if args.encode:
        print('Loading sequence to encode...')
        encode_seq = load_midi(args.encode)
        encode_seq = torch.from_numpy(encode_seq).long()

    latent = None
    if args.latent:
        print('Loading custom latent vector...')
        with open(args.latent) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                latent = np.array([float(x) for x in row])
                latent = torch.from_numpy(latent).float()

    print('=== Generating ===')

    fname = args.fname
    print('File: {}'.format(fname))
    generator = MusicGenerator(model)
    seq = generator.generate(args.length, encode_seq, args.temperature, latent)
    save_midi(fname, seq)

    if args.synth:
        data = synthesize(os.path.join(SAMPLES_DIR, fname + '.mid'))
        with open(os.path.join(SAMPLES_DIR, fname + '.mp3'), 'wb') as f:
            f.write(data)

if __name__ == '__main__':
    main()
