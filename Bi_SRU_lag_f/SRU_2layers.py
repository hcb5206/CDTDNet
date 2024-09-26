import torch.nn as nn
from Bi_SRU_lag_f import SRU_lag

"""
双层SRU
"""


class MultiLayerSRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiLayerSRU, self).__init__()
        self.sru1 = SRU_lag.Lstm(input_size, hidden_size)
        self.sru2 = SRU_lag.Lstm(hidden_size, hidden_size)

    def forward(self, x):
        _, out = self.sru1(x)
        out, hidden_all = self.sru2(out)
        return out, hidden_all
