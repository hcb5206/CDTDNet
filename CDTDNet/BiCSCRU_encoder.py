import torch
import torch.nn as nn
import CSCRU_Bi
import CSCRU_2Layers
"""
BiCSCRU_encoder
"""


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(BidirectionalLSTM, self).__init__()
        self.forward_lstm = CSCRU_Bi.Lstm(input_size, hidden_size, batch_size)
        self.backward_lstm = CSCRU_Bi.Lstm(input_size, hidden_size, batch_size)
        # self.bn = nn.BatchNorm1d(1)
        # self.forward_lstm = CSCRU_2Layers.MultiLayerLSTM(input_size, hidden_size)
        # self.backward_lstm = CSCRU_2Layers.MultiLayerLSTM(input_size, hidden_size)

    def forward(self, x):
        out_forward, cs_forward, fs_forward, hidden_att_forward = self.forward_lstm(x)
        # out_forward = self.bn(out_forward)
        x_reverse = torch.flip(x, [1])
        out_backward, cs_backward, fs_backward, hidden_att_backward = self.backward_lstm(x_reverse)
        hidden_att_backward = torch.flip(hidden_att_backward, [1])
        # out_backward = self.bn(out_backward)
        out = torch.cat((hidden_att_forward, hidden_att_backward), dim=2)
        out = out[:, -1:, :]

        return out, cs_forward, fs_forward, cs_backward, fs_backward, hidden_att_forward, hidden_att_backward
