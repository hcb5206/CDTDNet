import torch.nn as nn
import CSCRU

"""
2Layers_CSCRU
"""


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiLayerLSTM, self).__init__()
        self.lstm1 = CSCRU.Lstm(input_size, hidden_size)
        self.lstm2 = CSCRU.Lstm(hidden_size, hidden_size)

    def forward(self, x):
        _, out = self.lstm1(x)
        # print(out.shape)
        out, hidden_all = self.lstm2(out)
        return out, hidden_all
