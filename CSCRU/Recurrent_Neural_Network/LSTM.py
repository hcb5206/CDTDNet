import torch
import torch.nn as nn
import torch.nn.init as init

"""
LSTM
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_len = input_size + hidden_size
        self.batch_size = 64

        self.wg = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wi = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wf = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wo = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))

        self.bg = nn.Parameter(torch.Tensor(hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

        self.g = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.i = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.o = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.h = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.xc = torch.zeros(self.batch_size, 1, self.concat_len)

    def init_weights(self):

        init.xavier_uniform_(self.wg)
        init.xavier_uniform_(self.wi)
        init.xavier_uniform_(self.wf)
        init.xavier_uniform_(self.wo)

        init.constant_(self.bg, 0)
        init.constant_(self.bi, 0)
        init.constant_(self.bf, 0)
        init.constant_(self.bo, 0)

    def forward(self, x, s_prev=None, h_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        if s_prev is None:
            s_prev = torch.zeros_like(self.s)
        if h_prev is None:
            h_prev = torch.zeros_like(self.h)
        self.s_prev = s_prev
        self.h_prev = h_prev

        self.xc = torch.cat((x, self.h_prev), dim=2)
        self.g = torch.tanh(torch.matmul(self.xc, self.wg) + self.bg)
        self.i = torch.sigmoid(torch.matmul(self.xc, self.wi) + self.bi)
        self.f = torch.sigmoid(torch.matmul(self.xc, self.wf) + self.bf)
        self.o = torch.sigmoid(torch.matmul(self.xc, self.wo) + self.bo)
        self.s = self.g * self.i + self.s_prev * self.f
        self.h = torch.tanh(self.s) * self.o
        return self.s, self.h


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
        h_out = []
        for i in range(x.shape[1]):
            if i > 0:
                s_prev = self.lstm.s
                h_prev = self.lstm.h
            else:
                s_prev = None
                h_prev = None

            c, h = self.lstm(x[:, i, :][:, None, :], s_prev, h_prev)
            h_out.append(h)

        return h, torch.stack(h_out, dim=1).squeeze()
