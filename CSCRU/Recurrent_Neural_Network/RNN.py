import torch
import torch.nn as nn
import torch.nn.init as init

"""
RNN
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_len = input_size + hidden_size
        self.batch_size = 64

        self.w = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()
        self.h = torch.zeros(self.batch_size, 1, self.hidden_size)

    def init_weights(self):
        init.xavier_uniform_(self.w)
        init.constant_(self.b, 0)

    def forward(self, x, h_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        if h_prev is None:
            h_prev = torch.zeros_like(self.h)
        h_prev_in = h_prev
        xc = torch.cat((x, h_prev_in), dim=2)
        self.h = torch.tanh(torch.matmul(xc, self.w) + self.b)
        return self.h


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        self.lstm = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(self.lstm)

    def forward(self, x):
        assert x.dim() == 3
        for i in range(x.shape[1]):
            if i > 0:
                h_prev = self.lstm.h
            else:
                h_prev = None

            h = self.lstm(x[:, i, :][:, None, :], h_prev)
            prev_x = h

        return prev_x
