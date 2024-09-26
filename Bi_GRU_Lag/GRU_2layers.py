import torch.nn as nn
from Bi_GRU_Lag import GRU_lag

"""
双层GRU
"""


class MultiLayerSRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiLayerSRU, self).__init__()
        self.sru1 = GRU_lag.Lstm(input_size, hidden_size)
        self.sru2 = GRU_lag.Lstm(hidden_size, hidden_size)

    def forward(self, x):
        _, out = self.sru1(x)
        out, hidden_all = self.sru2(out)
        return out, hidden_all
