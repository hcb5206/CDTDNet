import torch
import torch.nn as nn
import torch.nn.init as init
from Attention_cst import Attention_Coupled

"""
SRU_decoder
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.w = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wr = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wf = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wx = nn.Parameter(torch.Tensor(self.input_size, hidden_size))

        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

        self.f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.r = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.h = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.xf = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.xr = torch.zeros(self.batch_size, 1, self.hidden_size)

    def init_weights(self):
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.wr)
        init.xavier_uniform_(self.wf)
        init.xavier_uniform_(self.wx)

        init.constant_(self.bf, 0)
        init.constant_(self.br, 0)

    def forward(self, x, spre, s_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        self.xc = x
        if s_prev is None:
            s_prev = spre

        self.s_prev = s_prev

        self.xf = torch.matmul(self.xc, self.w)
        self.xr = torch.matmul(self.xc, self.wx)
        self.f = torch.sigmoid(torch.matmul(self.xc, self.wf) + self.bf)
        self.r = torch.sigmoid(torch.matmul(self.xc, self.wr) + self.br)
        self.s = self.xf * (1 - self.f) + self.s_prev * self.f
        self.h = torch.tanh(self.s) * self.r + (1 - self.r) * self.xr
        return self.s, self.h


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, npred, seq_len, batch_size):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.npred = npred
        self.layers = nn.ModuleList()

        self.lstm = Lstmcell(self.input_size, self.hidden_size, self.batch_size)
        self.layers.append(self.lstm)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.ATT = Attention_Coupled(input_size=hidden_size, seq_len=seq_len)

    def forward(self, x, spre, hpre_att):
        assert x.dim() == 3
        output = []
        input_in = x
        for i in range(self.npred + 1):
            if i > 0:
                s_prev = self.lstm.s
            else:
                s_prev = None
            c, h = self.lstm(input_in, spre, s_prev)
            ATT_c = self.ATT(h, hpre_att)
            h_att = h + ATT_c
            prev_x = self.fc(h_att)
            input_in = prev_x
            if i > 0:
                output.append(prev_x)

        return torch.stack(output, dim=1).squeeze()
