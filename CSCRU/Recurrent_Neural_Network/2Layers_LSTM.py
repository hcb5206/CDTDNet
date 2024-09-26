import torch
import torch.nn as nn
import torch.nn.init as init

"""
2Layers_LSTM
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_len = input_size + hidden_size

        self.wg_s = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wi_s = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wf_s = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wo_s = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wg_f = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wi_f = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wf_f = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))
        self.wo_f = nn.Parameter(torch.Tensor(self.concat_len, hidden_size))

        self.bg_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bi_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bf_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bo_s = nn.Parameter(torch.Tensor(hidden_size))
        self.bg_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bi_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bf_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bo_f = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

        self.g_s = torch.zeros(64, 1, self.hidden_size)
        self.i_s = torch.zeros(64, 1, self.hidden_size)
        self.f_s = torch.zeros(64, 1, self.hidden_size)
        self.o_s = torch.zeros(64, 1, self.hidden_size)
        self.s_s = torch.zeros(64, 1, self.hidden_size)
        self.g_f = torch.zeros(64, 1, self.hidden_size)
        self.i_f = torch.zeros(64, 1, self.hidden_size)
        self.f_f = torch.zeros(64, 1, self.hidden_size)
        self.o_f = torch.zeros(64, 1, self.hidden_size)
        self.s_f = torch.zeros(64, 1, self.hidden_size)
        self.h = torch.zeros(64, 1, self.hidden_size)
        self.xc = torch.zeros(64, 1, self.concat_len)

    def init_weights(self):

        init.xavier_uniform_(self.wg_s)
        init.xavier_uniform_(self.wi_s)
        init.xavier_uniform_(self.wf_s)
        init.xavier_uniform_(self.wo_s)
        init.xavier_uniform_(self.wg_f)
        init.xavier_uniform_(self.wi_f)
        init.xavier_uniform_(self.wf_f)
        init.xavier_uniform_(self.wo_f)

        init.constant_(self.bg_s, 0)
        init.constant_(self.bi_s, 0)
        init.constant_(self.bf_s, 0)
        init.constant_(self.bo_s, 0)
        init.constant_(self.bg_f, 0)
        init.constant_(self.bi_f, 0)
        init.constant_(self.bf_f, 0)
        init.constant_(self.bo_f, 0)

    def forward(self, x, s_prev=None, f_prev=None, h_prev=None):
        assert x.shape[2] == self.input_size, 'input expect size:{},but get size:{}!!'.format(self.input_size,
                                                                                              x.shape[2])
        if s_prev is None:
            s_prev = torch.zeros_like(self.s_s)
        if f_prev is None:
            f_prev = torch.zeros_like(self.s_f)
        if h_prev is None:
            h_prev = torch.zeros_like(self.h)
        self.s_prev = s_prev
        self.f_prev = f_prev
        self.h_prev = h_prev

        self.xc = torch.cat((x, self.h_prev), dim=2)
        self.g_s = torch.tanh(torch.matmul(self.xc, self.wg_s) + self.bg_s)
        self.i_s = torch.sigmoid(torch.matmul(self.xc, self.wi_s) + self.bi_s)
        self.f_s = torch.sigmoid(torch.matmul(self.xc, self.wf_s) + self.bf_s)
        self.o_s = torch.sigmoid(torch.matmul(self.xc, self.wo_s) + self.bo_s)
        self.s_s = self.g_s * self.i_s + self.s_prev * self.f_s

        self.g_f = torch.tanh(torch.matmul(self.xc, self.wg_f) + self.bg_f)
        self.i_f = torch.sigmoid(torch.matmul(self.xc, self.wi_f) + self.bi_f)
        self.f_f = torch.sigmoid(torch.matmul(self.xc, self.wf_f) + self.bf_f)
        self.o_f = torch.sigmoid(torch.matmul(self.xc, self.wo_f) + self.bo_f)
        self.s_f = self.g_f * self.i_f + self.f_prev * self.f_f

        self.h = torch.tanh(self.s_s) * self.o_s + torch.tanh(self.s_f) * self.o_f
        return self.s_s, self.s_f, self.h


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        # for i in range(10):
        #     layer = Lstmcell(self.input_size, self.hidden_size)
        #     self.layers.append(layer)
        self.lstm = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(self.lstm)
        # self.lstm = self.layers[0]

    def forward(self, x):
        assert x.dim() == 3
        cs = []
        cf = []
        # s_prev_list = []
        # f_prev_list = []
        # h_prev_list = []
        # self.lstm_node_list = []
        for i in range(x.shape[1]):
            if i > 0:
                s_prev = self.lstm.s_s
                f_prev = self.lstm.s_f
                h_prev = self.lstm.h
            else:
                s_prev = None
                f_prev = None
                h_prev = None
            # s_prev_list.append(s_prev)
            # f_prev_list.append(f_prev)
            # h_prev_list.append(h_prev)
            c, f, h = self.lstm(x[:, i, :][:, None, :], s_prev, f_prev, h_prev)
            # self.lstm_node_list.append(Lstmcell(self.input_size, self.hidden_size))
            # self.lstm_node_list[i].forward(x[:, i, :][:, None, :], s_prev, f_prev, h_prev)
            # if i == 9:
            #     self.lstm_node_list = []
                # s_prev_list = []
                # f_prev_list = []
                # h_prev_list = []

            prev_x = h
            cs.append(c)
            cf.append(f)

        return prev_x, torch.stack(cs), torch.stack(cf)
