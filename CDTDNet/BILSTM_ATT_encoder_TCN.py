import torch
import torch.nn as nn
import torch.nn.init as init
import Attention_cst

"""
Bi-LSTM_6_ATT
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = 64

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

        # for i in range(10):
        self.lstm_f = Lstmcell(self.input_size, self.hidden_size)
        self.lstm_b = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(self.lstm_f)
        self.layers.append(self.lstm_b)
        # self.lstm = self.layers[0]

        # self.fc = nn.Linear(self.hidden_size, 1)
        self.ATT = Attention_cst.Feature_Attention(input_size=input_size, hidden_size=hidden_size, seq_len=10)
        # self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        # x = x.transpose(1, 2)
        assert x.dim() == 3
        x_reverse = torch.flip(x, [1])
        output_f = []
        output_b = []
        input_in = self.ATT(x[:, 0, :][:, None, :], torch.zeros((64, 1, self.hidden_size), dtype=torch.float32))
        input_in_reverse = self.ATT(x_reverse[:, 0, :][:, None, :],
                                    torch.zeros((64, 1, self.hidden_size), dtype=torch.float32))
        for i in range(x.shape[1]):
            if i > 0:
                s_prev1 = self.lstm_f.s
                f_prev1 = self.lstm_f.f
                s_prev2 = self.lstm_b.s
                f_prev2 = self.lstm_b.f
            else:
                s_prev1 = None
                f_prev1 = None
                s_prev2 = None
                f_prev2 = None
            # print(i, input_in, input_in.shape, '*' * 10)
            c1, f1, h1 = self.lstm_f(input_in, s_prev1, f_prev1)
            c2, f2, h2 = self.lstm_b(input_in_reverse, s_prev2, f_prev2)
            if i == x.shape[1] - 1:
                input_in = []
                input_in_reverse = []
            else:
                input_in = self.ATT(x[:, i + 1, :][:, None, :], h1)
                input_in_reverse = self.ATT(x_reverse[:, i + 1, :][:, None, :], h2)
            output_f.append(h1)
            output_b.append(h2)
        output = torch.cat(
            (torch.stack(output_f, dim=1).squeeze(), torch.flip(torch.stack(output_b, dim=1).squeeze(), [1])), dim=2)
        out = output[:, -1:, :]
        return out, c1, f1, c2, f2, torch.stack(output_f, dim=1).squeeze(), torch.flip(
            torch.stack(output_b, dim=1).squeeze(), [1])
