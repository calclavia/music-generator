import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    def __init__(self, input_size=1024, encoder_size=1024, decoder_size=1024, latent_size=1024):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.embd = nn.Embedding(NUM_ACTIONS, input_size)
        self.encoder = EncoderRNN(input_size, encoder_size, latent_size, 3)
        self.decoder = DecoderRNN(self.embd, input_size, latent_size, decoder_size, NUM_ACTIONS, 3)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        # Project to dense representation
        x = self.embd(x)
        # Encoder output is the latent vector
        mean, logvar, _ = self.encoder(x, hidden)
        std = torch.exp(0.5 * logvar)
        
        # Generate random latent vector
        z = Variable(torch.randn([batch_size, self.latent_size]))
        z = z.type(x.type())

        if x.is_cuda:
            z = z.cuda()

        z = z * std + mean

        decoder_output, _ = self.decoder(x[:, :-1], z)
        return decoder_output, mean, logvar


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.rnn = nn.GRU(input_size, hidden_size // 2, num_layers, batch_first=True, bidirectional=True)
        self.latent_projection = nn.Linear(hidden_size * num_layers, latent_size * 2)

    def forward(self, x, hidden=None):
        """
        Takes in a sequence of inputs and encodes them into mean and log variance.
        Return: (Mean, SD, Hidden)
        """
        _, hidden = self.rnn(x, hidden)
        
        # Project hidden state to latent vector
        x = torch.cat(list(hidden), dim=1)
        x = self.latent_projection(x)

        return x[:, :self.latent_size], x[:, self.latent_size:], hidden


class DecoderRNN(nn.Module):
    def __init__(self, embd, input_size, latent_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.latent_projection = nn.Linear(latent_size, hidden_size * num_layers)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Tie output embedding with input embd weights to improve regularization
        # https://arxiv.org/abs/1608.05859 https://arxiv.org/abs/1611.01462
        self.decoder = nn.Linear(hidden_size, output_size)
        self.decoder.weight = embd.weight

    def forward(self, x, latent=None, hidden=None):
        assert (latent is None and hidden is not None) or (hidden is None and latent is not None)
        # Project the latent vector to a size consumable by the GRU's memory
        if hidden is None:
            latent = self.latent_projection(latent)
            latent = latent.view(-1, self.num_layers, self.hidden_size)
            latent = latent.permute(1, 0, 2).contiguous()
            hidden = latent

        x, hidden = self.rnn(x, hidden)
        x = self.decoder(x)
        return x, hidden
