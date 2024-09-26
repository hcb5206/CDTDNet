import torch
import torch.nn as nn
import torch.nn.init as init
from Attention_cst import Attention_Coupled

"""
LSTM_6_decoder
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.w_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.w = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wf_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wf_f = nn.Parameter(torch.Tensor(self.hidden_size, hidden_size))
        self.wo_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wo_f = nn.Parameter(torch.Tensor(self.hidden_size, hidden_size))

        self.bf_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bf_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bo_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bo_f = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

        self.f_s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.f_f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.i_f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.o_s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.o_f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.h = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.xs = torch.zeros(self.batch_size, 1, self.hidden_size)

    def init_weights(self):
        init.xavier_uniform_(self.w_s)
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.wf_s)
        init.xavier_uniform_(self.wf_f)
        init.xavier_uniform_(self.wo_s)
        init.xavier_uniform_(self.wo_f)

        init.constant_(self.bf_s, 0)
        init.constant_(self.bf_f, 0)
        init.constant_(self.bo_s, 0)
        init.constant_(self.bo_f, 0)

    def forward(self, x, sprev, fprev, s_prev=None, f_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        self.xc = x

        if s_prev is None:
            s_prev = sprev
        if f_prev is None:
            f_prev = fprev
        self.s_prev = s_prev
        self.f_prev = f_prev

        self.xs = torch.matmul(self.xc, self.w_s)
        self.f_s = torch.sigmoid(torch.matmul(self.xc, self.wf_s) + self.bf_s)
        self.f_f = torch.sigmoid(torch.matmul(self.s_prev, self.wf_f) + self.bf_f)
        self.i_f = self.s_prev * (1 - self.f_s)
        self.o_s = torch.sigmoid(torch.matmul(self.xc, self.wo_s) + self.bo_s)
        self.o_f = torch.sigmoid(torch.matmul(self.s_prev, self.wo_f) + self.bo_f)

        self.s = self.xs * (1 - self.f_s) + self.s_prev * self.f_s
        self.f = self.s_prev * (1 - self.f_s) + self.f_prev * self.f_f

        self.h = torch.tanh(self.s) * self.o_s + torch.tanh(self.f) * self.o_f + (torch.matmul(self.xc, self.w))

        return self.s, self.f, self.h


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

    def forward(self, x, spre, fpre, hpre_att):
        assert x.dim() == 3
        output = []
        input_in = x
        for i in range(self.npred):
            if i > 0:
                s_prev = self.lstm.s
                f_prev = self.lstm.f
            else:
                s_prev = None
                f_prev = None
            c, f, h = self.lstm(input_in, spre, fpre, s_prev, f_prev)
            ATT_c = self.ATT(h, hpre_att)
            h_att = h + ATT_c
            prev_x = self.fc(h_att)
            input_in = prev_x
            output.append(prev_x)
        return torch.stack(output, dim=1).squeeze()
