import torch
import torch.nn as nn
import CSCRU_Bi

"""
2Layers_CSCRU_encoder
"""


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(MultiLayerLSTM, self).__init__()
        self.lstm1 = CSCRU_Bi.Lstm(input_size, hidden_size, batch_size)
        self.lstm2 = CSCRU_Bi.Lstm(hidden_size, hidden_size, batch_size)

    def forward(self, x):
        _, _, _, out = self.lstm1(x)
        out, cs, fs, h_att = self.lstm2(out)
        return out, cs, fs, h_att
