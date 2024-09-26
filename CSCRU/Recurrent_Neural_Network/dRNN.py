import torch
import torch.nn as nn
import torch.nn.init as init

"""
dRNN
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_len = input_size + hidden_size * 2
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

    def forward(self, x, s_prev=None, s_d_prev=None, h_prev=None, h_d_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        if s_prev is None:
            s_prev = torch.zeros_like(self.s)
        if s_d_prev is None:
            s_d_prev = torch.zeros_like(self.s)
        if h_prev is None:
            h_prev = torch.zeros_like(self.h)
        if h_d_prev is None:
            h_d_prev = torch.zeros_like(self.h)
        self.s_prev = s_prev
        self.s_d_prev = s_d_prev
        self.h_prev = h_prev
        self.h_d_prev = h_d_prev

        self.xc = torch.cat((x, self.h_prev, self.h_d_prev), dim=2)
        self.g = torch.tanh(torch.matmul(self.xc, self.wg) + self.bg)
        self.i = torch.sigmoid(torch.matmul(self.xc, self.wi) + self.bi)
        self.f = torch.sigmoid(torch.matmul(self.xc, self.wf) + self.bf)
        self.o = torch.sigmoid(torch.matmul(self.xc, self.wo) + self.bo)
        self.s = self.i * (self.f * self.s_prev + (1 - self.f) * self.s_d_prev) + (1 - self.i) * self.g
        self.h = self.s * self.o
        return self.s, self.h


class dRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, ds):
        super(dRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ds = ds
        self.layers = nn.ModuleList()

        self.lstm = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(self.lstm)

    def forward(self, x):
        assert x.dim() == 3
        h = None
        c_out = []
        h_out = []
        for i in range(x.shape[1]):
            if i < self.ds:
                s_d_prev = None
                h_d_prev = None
            else:
                s_d_prev = c_out[i - self.ds]
                h_d_prev = h_out[i - self.ds]
            if i > 0:
                s_prev = self.lstm.s
                h_prev = self.lstm.h
            else:
                s_prev = None
                h_prev = None

            c, h = self.lstm(x[:, i, :][:, None, :], s_prev, s_d_prev, h_prev, h_d_prev)
            h_out.append(h)
            c_out.append(c)
        return torch.stack(h_out, dim=1).squeeze(), h


class dRNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, ds):
        super(dRNNBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if not isinstance(ds, list):
            raise ValueError("dilation must be a list of integers representing the dilation for each layer.")

        self.num_layers = len(ds)
        self.dilation = ds

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(dRNNLayer(self.input_size, self.hidden_size, self.dilation[i]))
            else:
                self.layers.append(dRNNLayer(self.hidden_size, self.hidden_size, self.dilation[i]))

    def forward(self, x):
        hidden_all = None
        h = None
        for i in range(self.num_layers):
            hidden_all, h = self.layers[i](x)
            x = hidden_all
        return hidden_all, h


class dRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super(dRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if not isinstance(dilation, list) or not all(isinstance(d, list) for d in dilation):
            raise ValueError("dilation must be a list of lists, where each inner list contains integers.")

        self.blocks = nn.ModuleList()
        for dilations in dilation:
            self.blocks.append(dRNNBlock(self.input_size, self.hidden_size, dilations))
            self.input_size = self.hidden_size

    def forward(self, x):
        hidden_all = None
        h = None
        i = 0
        for block in self.blocks:
            hidden_all, h = block(x)
            if i > 0:
                x = hidden_all + x
            else:
                x = hidden_all
            i = i + 1
        return h, hidden_all
