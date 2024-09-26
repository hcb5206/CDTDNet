import torch
import torch.nn as nn
import torch.nn.init as init
from Attention_cst import Attention_Coupled

"""
2Layers_Coupled_attention
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # self.concat_len = input_size + hidden_size

        self.w_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.w = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        # self.w_f = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wf_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wf_f = nn.Parameter(torch.Tensor(self.hidden_size, hidden_size))
        self.wo_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wo_f = nn.Parameter(torch.Tensor(self.hidden_size, hidden_size))
        # self.wf_f = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        # self.wx = nn.Parameter(torch.Tensor(self.input_size, hidden_size))

        # self.bg = nn.Parameter(torch.Tensor(hidden_size))
        # self.bi = nn.Parameter(torch.Tensor(hidden_size))
        # self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.bf_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bf_f = nn.Parameter(torch.Tensor(hidden_size))
        # self.bf_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bo_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bo_f = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

        # self.g = torch.zeros(64, 1, self.hidden_size)
        # self.i = torch.zeros(64, 1, self.hidden_size)
        self.f_s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.f_f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.i_f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.o_s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.o_f = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.s = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.h = torch.zeros(self.batch_size, 1, self.hidden_size)
        self.f = torch.zeros(self.batch_size, 1, self.hidden_size)
        # self.xf = torch.zeros(64, 1, self.hidden_size)
        self.xs = torch.zeros(self.batch_size, 1, self.hidden_size)

    def init_weights(self):
        init.xavier_uniform_(self.w_s)
        init.xavier_uniform_(self.w)
        # init.xavier_uniform_(self.w_f)
        init.xavier_uniform_(self.wf_s)
        init.xavier_uniform_(self.wf_f)
        init.xavier_uniform_(self.wo_s)
        init.xavier_uniform_(self.wo_f)
        # init.xavier_uniform_(self.wf_f)
        # init.xavier_uniform_(self.wx)

        # init.constant_(self.bg, 0)
        # init.constant_(self.bi, 0)
        # init.constant_(self.b, 0)
        init.constant_(self.bf_s, 0)
        init.constant_(self.bf_f, 0)
        # init.constant_(self.bf_f, 0)
        init.constant_(self.bo_s, 0)
        init.constant_(self.bo_f, 0)

    def forward(self, x, sprev, fprev, s_prev=None, f_prev=None):
        # print(self.input_size)
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        self.xc = x

        if s_prev is None:
            s_prev = sprev
        if f_prev is None:
            f_prev = fprev
        # if h_prev is None:
        #     h_prev = torch.zeros_like(self.h)
        self.s_prev = s_prev
        self.f_prev = f_prev
        # self.h_prev = h_prev

        # self.xc = torch.cat((x, self.h_prev), dim=2)
        self.xs = torch.matmul(self.xc, self.w_s)
        # self.xf = torch.matmul(self.xc, self.w_f)
        # self.xr = torch.matmul(self.xc, self.wx)
        # self.g = torch.tanh(torch.matmul(self.xc, self.wg) + self.bg)
        # self.i = torch.sigmoid(torch.matmul(self.xc, self.wi) + self.bi)
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
        self.npred = npred
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.layers = nn.ModuleList()

        # for i in range(10):
        self.lstm1 = Lstmcell(self.input_size, self.hidden_size, self.batch_size)
        self.lstm2 = Lstmcell(self.hidden_size, self.hidden_size, self.batch_size)
        self.layers.append(self.lstm1)
        self.layers.append(self.lstm2)
        # self.lstm = self.layers[0]

        self.fc = nn.Linear(self.hidden_size, 1)
        self.ATT = Attention_Coupled(input_size=hidden_size, seq_len=seq_len)
        # self.bn = nn.BatchNorm1d(1)

    def forward(self, x, spre, fpre, hpre_att):
        assert x.dim() == 3
        output = []
        input_in = x
        for i in range(self.npred):
            if i > 0:
                s_prev1 = self.lstm1.s
                f_prev1 = self.lstm1.f
                s_prev2 = self.lstm2.s
                f_prev2 = self.lstm2.f
            else:
                s_prev1 = None
                f_prev1 = None
                s_prev2 = None
                f_prev2 = None
            # print(i, input_in, input_in.shape, '*' * 10)
            c1, f1, h1 = self.lstm1(input_in, spre, fpre, s_prev1, f_prev1)
            c2, f2, h2 = self.lstm2(h1, spre, fpre, s_prev2, f_prev2)
            # print(h, h.shape)
            ATT_c = self.ATT(h2, hpre_att)
            h_att = h2 + ATT_c
            prev_x = self.fc(h_att)
            input_in = prev_x
            # input_in = input_in.transpose(1, 2)
            # input_in = self.bn(input_in)
            # input_in = input_in.transpose(1, 2)
            output.append(prev_x)
            # print(i, prev_x, '*' * 10)
            # print(output, len(output), '*' * 100)
        return torch.stack(output, dim=1).squeeze()
