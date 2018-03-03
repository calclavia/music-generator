import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=512, num_layers=4, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.style_units = style_units

        # RNN
        self.rnns = [
            nn.GRU(NUM_ACTIONS + style_units, num_units, 2, batch_first=True) if i == 0 else 
            DilatedRNN(nn.GRU(num_units, num_units, batch_first=True), 2 ** i)
            for i in range(num_layers)
        ]

        self.output_linear = nn.Linear(self.num_units, NUM_ACTIONS)

        for i, rnn in enumerate(self.rnns):
            self.add_module('rnn_' + str(i), rnn)

        # Style
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        # self.style_layer = nn.Linear(self.style_units, self.num_units * self.num_layers)

    def forward(self, x, style, states=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Distributed style representation
        style = self.style_linear(style)
        # style = F.tanh(self.style_layer(style))
        style = style.unsqueeze(1).expand(batch_size, seq_len, self.style_units)
        x = torch.cat((x, style), dim=2)

        ## Process RNN ##
        if states is None:
            states = [None for _ in range(self.num_layers)]

        for l, rnn in enumerate(self.rnns):
            prev_x = x
            x, states[l] = rnn(x, states[l])

            if l > 0:
                x = prev_x + x

        x = self.output_linear(x)
        return x, states

    def generate(self, x, style, states, temperature=1):
        """ Returns the probability of outputs """
        x, states = self.forward(x, style, states)
        seq_len = x.size(1)
        x = x.view(-1, NUM_ACTIONS)
        x = F.softmax(x / temperature, dim=1)
        x = x.view(-1, seq_len, NUM_ACTIONS)
        return x, states

class DilatedRNN(nn.Module):
    """ https://arxiv.org/pdf/1710.02224.pdf """
    def __init__(self, wrap_rnn, dilation=1):
        """
        Args:
            wrap_rnn: The RNN module to wrap.
            dilation: The dilation factor
        """
        super().__init__()
        assert wrap_rnn.batch_first
        self.rnn = wrap_rnn
        self.dilation = dilation
    
    def forward(self, x, states):
        """
        Args:
            x: A sequence of features [batch, seq_len, features]
        """
        # The number of dilation = the number of parallelism that can be achieved.
        # Move the additional parallels into batch dimension
        batch_size = x.size(0)
        seq_len = x.size(1)
        if seq_len == 1:
            # Single step requires us to store which step we are on.
            if states is None:
                # Each memory tensor corresponds to a dilation.
                states = (0, tuple(None for _ in range(self.dilation)))
            step, memories = states
            memory_id = step % self.dilation
            x, memory = self.rnn(x, memories[memory_id])
            states = (step + 1, memories[:memory_id] + (memory,) + memories[memory_id + 1:])
            return x, states

        # Taking in a full sequence
        assert seq_len % self.dilation == 0, seq_len
        x = x.unfold(1, self.dilation, self.dilation)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size * self.dilation, seq_len // self.dilation, -1)
        x, states = self.rnn(x, states)
        # X is now [batch * dilation, seq_len//dilation, features]
        # We want to restore it back into [batch, seq_len, features]
        # But we can't simply reshape it, because that messes up the order.
        x = x.contiguous().view(batch_size, self.dilation, seq_len // self.dilation, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, seq_len, -1)
        return x, states
