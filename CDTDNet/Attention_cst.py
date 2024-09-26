import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_size = input_size

        self.linear_s_1 = nn.Linear(self.seq_len, self.hidden_size)
        self.linear_s_2 = nn.Linear(self.hidden_size, self.seq_len, bias=False)
        # self.linear_t_1 = nn.Linear(self.input_size, self.hidden_size)
        # self.linear_t_2 = nn.Linear(self.hidden_size, self.input_size, bias=False)

        self.tanh_s = nn.Tanh()
        # self.tanh_t = nn.Tanh()

        self.sigmoid_s = nn.Sigmoid()
        # self.sigmoid_t = nn.Sigmoid()

        self.softmax_s = nn.Softmax(dim=1)
        # self.softmax_t = nn.Softmax(dim=1)

        self.dropout_s = nn.Dropout1d(p=0.5)
        # self.dropout_t = nn.Dropout1d(p=0.5)

        self.bn_s = nn.BatchNorm1d(self.input_size)
        # self.bn_t = nn.BatchNorm1d(self.seq_len)

    def forward(self, x):
        x_s = x
        s_s = self.linear_s_1(x_s)
        s_s = self.bn_s(s_s)
        s_s = self.tanh_s(s_s)
        s_s = self.dropout_s(s_s)
        s_s = self.linear_s_2(s_s)
        s_s = self.softmax_s(s_s)

        # x_t = x.transpose(1, 2)
        # s_t = self.linear_t_1(x_t)
        # s_t = self.bn_t(s_t)
        # s_t = self.tanh_t(s_t)
        # s_t = self.dropout_t(s_t)
        # s_t = self.linear_t_2(s_t)
        # s_t = self.softmax_t(s_t)
        # s_t = s_t.transpose(1, 2)
        x_att = x + x * s_s

        return x_att


class Feature_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super(Feature_Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.linear_1 = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.linear_2 = nn.Linear(1, self.seq_len, bias=False)
        self.linear_3 = nn.Linear(1, self.seq_len, bias=False)
        self.linear_4 = nn.Linear(self.seq_len, 1, bias=False)

        self.tanh = nn.Tanh()

        self.softmax_s = nn.Softmax(dim=1)

    def forward(self, x, x_hidden):
        x_hidden_1 = self.linear_1(x_hidden).transpose(1, 2)
        x_hidden_2 = self.linear_2(x_hidden_1)
        x_1 = self.linear_3(x.transpose(1, 2))
        x_s = self.linear_4(self.tanh(x_hidden_2 + x_1))
        x_soft = self.softmax_s(x_s).transpose(1, 2)
        x_att = x_soft * x

        return x_att


class Attention_Coupled(nn.Module):
    def __init__(self, input_size, seq_len):
        super(Attention_Coupled, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.fc1 = nn.Linear(input_size, input_size, bias=False)
        self.fc2 = nn.Linear(1, input_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(seq_len)
        self.dropout = nn.Dropout1d(p=0.5)

    def forward(self, x_h, x_hpre):
        ATT_out = torch.zeros_like(x_hpre[:, -1:, :])
        x_hpre_s = self.fc1(x_hpre)
        x_hpre_s = self.bn(x_hpre_s)
        x_hpre_s = self.tanh(x_hpre_s)
        x_hpre_s = self.dropout(x_hpre_s)
        x_h_s = x_h.transpose(1, 2)
        a_t = torch.matmul(x_hpre_s, x_h_s)
        a_s = self.softmax(a_t)
        ATT = self.fc2(a_s)
        for i in range(ATT.shape[1]):
            ATT_s = ATT[:, i, :][:, None, :] * x_hpre[:, i, :][:, None, :]
            ATT_out += ATT_s
        return ATT_out


class AttentionMechanism(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length):
        super(AttentionMechanism, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.u = nn.Parameter(torch.randn(hidden_size, sequence_length))
        self.w = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, h):
        # print(self.u, '*' * 100, self.w, '*' * 100, self.b, '*' * 100)
        a1 = torch.matmul(h, self.w)
        a2 = a1 + self.b
        a3 = torch.tanh(a2)
        a4 = torch.matmul(a3, self.u)
        a = torch.softmax(a4, dim=-1)
        S = torch.matmul(a.transpose(1, 2), h)

        return S


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_query = nn.Linear(input_dim, hidden_dim)
        self.W_key = nn.Linear(input_dim, hidden_dim)
        self.W_value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Calculate query, key, and value
        x = x.permute(0, 2, 1)
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to value to get weighted sum
        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1)

        return output


class SelfAttention_step(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention_step, self).__init__()

        self.hidden_dim = hidden_dim
        self.W_query = nn.Linear(input_dim, hidden_dim)
        self.W_key = nn.Linear(input_dim, hidden_dim)
        self.W_value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Calculate query, key, and value

        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to value to get weighted sum
        output = torch.matmul(attention_weights, value)

        return output
