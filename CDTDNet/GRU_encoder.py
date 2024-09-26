import torch
import torch.nn as nn
import torch.nn.init as init

"""
GRU_encoder
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.concat_len = input_size + hidden_size

        self.wr = nn.Parameter(torch.Tensor(self.concat_len, self.hidden_size))
        self.wz = nn.Parameter(torch.Tensor(self.concat_len, self.hidden_size))
        self.whh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.whx = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))

        self.br = nn.Parameter(torch.Tensor(hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))
        self.bh = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

        self.r = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.z = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.g = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.h = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.xc = torch.zeros(self.batch_size, 1, self.concat_len)

    def init_weights(self):
        init.xavier_uniform_(self.wr)
        init.xavier_uniform_(self.wz)
        init.xavier_uniform_(self.whh)
        init.xavier_uniform_(self.whx)

        init.constant_(self.br, 0)
        init.constant_(self.bz, 0)
        init.constant_(self.bh, 0)

    def forward(self, x, h_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        self.xl = x
        if h_prev is None:
            h_prev = torch.zeros_like(self.h)
        self.h_prev = h_prev

        self.xc = torch.cat((x, self.h_prev), dim=2)
        self.r = torch.sigmoid(torch.matmul(self.xc, self.wr) + self.br)
        self.z = torch.sigmoid(torch.matmul(self.xc, self.wz) + self.bz)
        self.g = torch.tanh(torch.matmul(self.r * self.h_prev, self.whh) + torch.matmul(self.xl, self.whx) + self.bh)
        self.h = (1 - self.z) * self.h_prev + self.z * self.g
        return self.h


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = nn.ModuleList()

        self.lstm = Lstmcell(self.input_size, self.hidden_size, self.batch_size)
        self.layers.append(self.lstm)

    def forward(self, x):
        assert x.dim() == 3
        h_all = []
        for i in range(x.shape[1]):
            if i > 0:
                h_prev = self.lstm.h
            else:
                h_prev = None
            h = self.lstm(x[:, i, :][:, None, :], h_prev)
            h_all.append(h)

        return h, torch.stack(h_all, dim=1).squeeze()
