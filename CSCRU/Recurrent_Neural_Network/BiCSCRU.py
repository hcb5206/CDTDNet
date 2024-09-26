import torch
import torch.nn as nn
import CSCRU
import CSCRU_2Layers
"""
Bi-CSCRU
"""


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BidirectionalLSTM, self).__init__()
        self.forward_lstm = LSTM_6.Lstm(input_size, hidden_size)
        self.backward_lstm = LSTM_6.Lstm(input_size, hidden_size)
        # self.forward_lstm = LSTM_10.MultiLayerLSTM(input_size, hidden_size)
        # self.backward_lstm = LSTM_10.MultiLayerLSTM(input_size, hidden_size)
        # self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        _, out_forward = self.forward_lstm(x)
        x_reverse = torch.flip(x, [1])
        _, out_backward = self.backward_lstm(x_reverse)
        out_backward = torch.flip(out_backward, [1])
        # print(out_forward.shape, out_backward.shape)
        out = torch.cat((out_forward, out_backward), dim=2)
        # out = self.fc(out)

        return out
