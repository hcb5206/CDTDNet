import torch
import torch.nn as nn
import torch.nn.init as init

"""
LSTM_补充门_并行_带遗忘细胞_decoder
"""


class Lstmcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstmcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = 64
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
    def __init__(self, input_size, hidden_size, npred):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.npred = npred
        self.layers = nn.ModuleList()

        # for i in range(10):
        self.lstm = Lstmcell(self.input_size, self.hidden_size)
        self.layers.append(self.lstm)
        # self.lstm = self.layers[0]

        self.fc = nn.Linear(self.hidden_size, 1)
        # self.droupout = nn.Dropout(p=0.2)

    def forward(self, x, spre, fpre):
        assert x.dim() == 3
        output = []
        input_in = x
        # cs = []
        # fs = []
        # s_prev_list = []
        # h_prev_list = []
        # self.lstm_node_list = []
        for i in range(self.npred + 1):
            if i > 0:
                s_prev = self.lstm.s
                f_prev = self.lstm.f
                # h_prev = self.lstm_node_list[-1].h
            else:
                s_prev = None
                f_prev = None
                # h_prev = None
            # s_prev_list.append(s_prev)
            # h_prev_list.append(h_prev)
            # print(input_in)
            c, f, h = self.lstm(input_in, spre, fpre, s_prev, f_prev)
            # self.lstm_node_list.append(Lstmcell(self.input_size, self.hidden_size))
            # self.lstm_node_list[i].forward(input_in, spre, fpre, s_prev, f_prev)
            # if i == self.npred:
            #     self.lstm_node_list = []
                # s_prev_list = []
                # h_prev_list = []
            prev_x = self.fc(h)
            # print(prev_x)
            # if train_tru:
            #     if i == 5:
            #         input_in = []
            #     else:
            #         teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            #         # print(teacher_forcing)
            #         input_in = trg[:, i, :][:, None, :] if teacher_forcing else prev_x
            #         # print(trg[:, i, :][:, None, :])
            # else:
            input_in = prev_x
            if i > 0:
                output.append(prev_x)
            # print(i, input_in, '*' * 10)
            # print(i, prev_x, '*' * 10)
            # print(output, len(output), '*' * 100)
        return torch.stack(output, dim=1).squeeze()
