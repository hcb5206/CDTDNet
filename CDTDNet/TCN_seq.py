import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init
import Attention_cst


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class AttentionMechanism(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length):
        super(AttentionMechanism, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.u = nn.Parameter(torch.randn(hidden_size, sequence_length))
        self.w = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, h):
        a1 = torch.matmul(h, self.w)
        a2 = a1 + self.b
        a3 = torch.tanh(a2)
        a4 = torch.matmul(a3, self.u)
        # e = torch.matmul(torch.tanh(torch.matmul(h, self.w) + self.b), self.u)

        a = torch.softmax(a4, dim=1)

        S = torch.matmul(a.transpose(1, 2), h)

        return S


class SelfAttention_step(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention_step, self).__init__()

        self.hidden_dim = hidden_dim
        self.W_query = nn.Linear(input_dim, hidden_dim)
        self.W_key = nn.Linear(input_dim, hidden_dim)
        self.W_value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):

        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32))

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)

        return output


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, init=True):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print(f"TemporalBlock:x.shape:{x.shape}")
        out = self.net(x)
        # print(out.shape)
        # print(f"TemporalBlock:out.shape:{out.shape}")
        res = x if self.downsample is None else self.downsample(x)
        # print(f"TemporalBlock:res.shape:{res.shape}")
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

    def forward(self, x):
        output = self.tcn(x)
        return output
