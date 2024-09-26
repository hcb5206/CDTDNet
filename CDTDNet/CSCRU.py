import torch
import torch.nn as nn
import torch.nn.init as init

"""
CSCRU
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = 64  # Air:64, Energy:128, NFLX:64, Traffic: 128

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

    def forward(self, x, s_prev=None, f_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        self.xc = x

        if s_prev is None:
            s_prev = torch.zeros_like(self.s)
        if f_prev is None:
            f_prev = torch.zeros_like(self.f)

        self.s_prev = s_prev
        self.f_prev = f_prev

        self.xs = torch.matmul(self.xc, self.w_s)
        self.f_s = torch.sigmoid(torch.matmul(self.xc, self.wf_s) + self.bf_s)
        self.f_f = torch.sigmoid(torch.matmul(self.s_prev, self.wf_f) + self.bf_f)
        self.o_s = torch.sigmoid(torch.matmul(self.xc, self.wo_s) + self.bo_s)
        self.o_f = torch.sigmoid(torch.matmul(self.s_prev, self.wo_f) + self.bo_f)

        self.s = self.xs * (1 - self.f_s) + self.s_prev * self.f_s
        self.f = self.s_prev * (1 - self.f_s) + self.f_prev * self.f_f

        self.h = torch.tanh(self.s) * self.o_s + torch.tanh(self.f) * self.o_f + torch.matmul(self.xc, self.w)
        return self.s, self.f, self.h


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
                f_prev = self.lstm.f
            else:
                s_prev = None
                f_prev = None

            c, f, h = self.lstm(x[:, i, :][:, None, :], s_prev, f_prev)
            h_out.append(h)

        # return h, torch.stack(h_out, dim=1).squeeze()
        return h, torch.stack(h_out, dim=1).squeeze().unsqueeze(dim=1)
