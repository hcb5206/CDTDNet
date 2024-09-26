import torch
import torch.nn as nn
import torch.nn.init as init

"""
Bi-SRU
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.concat_len = input_size + hidden_size

        self.w_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        # self.w_f = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wr_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wr_f = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wf_s = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        # self.wf_f = nn.Parameter(torch.Tensor(self.input_size, hidden_size))
        self.wx = nn.Parameter(torch.Tensor(self.input_size, hidden_size))

        # self.bg = nn.Parameter(torch.Tensor(hidden_size))
        # self.bi = nn.Parameter(torch.Tensor(hidden_size))
        self.bf_s = nn.Parameter(torch.Tensor(hidden_size))
        # self.bf_f = nn.Parameter(torch.Tensor(hidden_size))
        self.br_s = nn.Parameter(torch.Tensor(hidden_size))
        self.br_f = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

        # self.g = torch.zeros(64, 1, self.hidden_size)
        # self.i = torch.zeros(64, 1, self.hidden_size)
        self.f_s = torch.zeros(64, 1, self.hidden_size)
        self.f_f = torch.zeros(64, 1, self.hidden_size)
        self.r_s = torch.zeros(64, 1, self.hidden_size)
        self.r_f = torch.zeros(64, 1, self.hidden_size)
        self.s = torch.zeros(64, 1, self.hidden_size)
        self.h = torch.zeros(64, 1, self.hidden_size)
        self.f = torch.zeros(64, 1, self.hidden_size)
        # self.xf = torch.zeros(64, 1, self.hidden_size)
        self.xr = torch.zeros(64, 1, self.hidden_size)

    def init_weights(self):
        init.xavier_uniform_(self.w_s)
        # init.xavier_uniform_(self.w_f)
        init.xavier_uniform_(self.wr_s)
        init.xavier_uniform_(self.wr_f)
        init.xavier_uniform_(self.wf_s)
        # init.xavier_uniform_(self.wf_f)
        init.xavier_uniform_(self.wx)

        # val_range = (3.0 / self.input_size) ** 0.5
        # self.w_s.data.uniform_(-val_range, val_range)
        # self.wr_s.data.uniform_(-val_range, val_range)
        # self.wr_f.data.uniform_(-val_range, val_range)
        # self.wf_s.data.uniform_(-val_range, val_range)
        # self.wx.data.uniform_(-val_range, val_range)

        # init.constant_(self.bg, 0)
        # init.constant_(self.bi, 0)
        init.constant_(self.bf_s, 0)
        # init.constant_(self.bf_f, 0)
        init.constant_(self.br_s, 0)
        init.constant_(self.br_f, 0)

        # self.bf_s.data.zero_()
        # self.br_s.data.zero_()
        # self.br_f.data.zero_()
        # bias_val, hidden_size = 0, self.hidden_size
        # self.bf_s.data[hidden_size:].zero_().add_(bias_val)
        # self.br_s.data[hidden_size:].zero_().add_(bias_val)
        # self.br_f.data[hidden_size:].zero_().add_(bias_val)

    def forward(self, x, s_prev=None, f_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        self.xc = x

        if s_prev is None:
            s_prev = torch.zeros_like(self.s)
        if f_prev is None:
            f_prev = torch.zeros_like(self.f)
        # if h_prev is None:
        #     h_prev = torch.zeros_like(self.h)
        self.s_prev = s_prev
        self.f_prev = f_prev
        # self.h_prev = h_prev

        # self.xc = torch.cat((x, self.h_prev), dim=2)
        self.xs = torch.matmul(self.xc, self.w_s)
        # self.xf = torch.matmul(self.xc, self.w_f)
        self.xr = torch.matmul(self.xc, self.wx)
        # self.g = torch.tanh(torch.matmul(self.xc, self.wg) + self.bg)
        # self.i = torch.sigmoid(torch.matmul(self.xc, self.wi) + self.bi)
        self.f_s = torch.sigmoid(torch.matmul(self.xc, self.wf_s) + self.bf_s)
        self.f_f = 1 - self.f_s
        self.r_s = torch.sigmoid(torch.matmul(self.xc, self.wr_s) + self.br_s)
        self.r_f = torch.sigmoid(torch.matmul(self.xc, self.wr_f) + self.br_f)

        self.s = self.xs * (1 - self.f_s) + self.s_prev * self.f_s
        self.f = self.xs * (self.f_s) + self.f_prev * self.f_f

        self.h = torch.tanh(self.s) * self.r_s + (1 - self.r_s) * self.xr + torch.tanh(self.f) * self.r_f + (
                1 - self.r_f) * self.xr
        return self.s, self.f, self.h


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        # for i in range(10):
        layer_f = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(layer_f)
        layer_b = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(layer_b)
        self.lstm_f = self.layers[0]
        self.lstm_b = self.layers[1]

    def forward(self, x):
        assert x.dim() == 3
        cs_f = []
        fs_f = []
        cs_b = []
        fs_b = []
        # s_prev_list = []
        # h_prev_list = []
        self.lstm_node_list_f = []
        self.lstm_node_list_b = []
        for i in range(x.shape[1]):
            if len(self.lstm_node_list_f) > 0:
                s_prev_f = self.lstm_node_list_f[-1].s
                f_prev_f = self.lstm_node_list_f[-1].f
                s_prev_b = self.lstm_node_list_b[-1].s
                f_prev_b = self.lstm_node_list_b[-1].f
                # h_prev = self.lstm_node_list[-1].h
            else:
                s_prev_f = None
                f_prev_f = None
                s_prev_b = None
                f_prev_b = None
                # h_prev = None
            # s_prev_list.append(s_prev)
            # h_prev_list.append(h_prev)
            c_f, f_f, h_f = self.lstm_f(x[:, i, :][:, None, :], s_prev_f, f_prev_f)
            c_b, f_b, h_b = self.lstm_b(x[:, 9 - i, :][:, None, :], s_prev_b, f_prev_b)
            self.lstm_node_list_f.append(Lstmcell(self.input_size, self.hidden_size))
            self.lstm_node_list_b.append(Lstmcell(self.input_size, self.hidden_size))
            self.lstm_node_list_f[i].forward(x[:, i, :][:, None, :], s_prev_f, f_prev_f)
            self.lstm_node_list_b[i].forward(x[:, 9 - i, :][:, None, :], s_prev_b, f_prev_b)
            if i == 9:
                self.lstm_node_list_f = []
                self.lstm_node_list_b = []
                # s_prev_list = []
                # h_prev_list = []

            # print(h_f.shape, h_b.shape)
            prev_x = torch.cat((h_f, h_b), dim=2)
            cs_f.append(c_f)
            fs_f.append(f_f)
            cs_b.append(c_b)
            fs_b.append(f_b)

        return prev_x, torch.stack(cs_f), torch.stack(fs_f), torch.stack(cs_b), torch.stack(fs_b)
