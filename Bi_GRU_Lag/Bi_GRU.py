import torch
import torch.nn as nn
# import GRU_lag
from Bi_GRU_Lag import GRU_2layers
"""
双向SRU
"""


class BidirectionalSRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BidirectionalSRU, self).__init__()
        # self.forward_lstm = LSTM_10.MultiLayerLSTM(input_size, hidden_size)
        # self.backward_lstm = LSTM_10.MultiLayerLSTM(input_size, hidden_size)
        self.forward_sru = GRU_2layers.MultiLayerSRU(input_size, hidden_size)
        self.backward_sru = GRU_2layers.MultiLayerSRU(input_size, hidden_size)
        # self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        _, out_forward = self.forward_sru(x)
        x_reverse = torch.flip(x, [1])
        _, out_backward = self.backward_sru(x_reverse)
        out_backward = torch.flip(out_backward, [1])
        # print(out_forward.shape, out_backward.shape)
        out = torch.cat((out_forward, out_backward), dim=2)
        # out = self.fc(out)

        return out
