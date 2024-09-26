import pandas as pd
import numpy as np
import torch
import os
import time
# import keras
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import Attention_cst
import LSTM
import LSTM_1
import LSTM_2
import Peephole_LSTM
import CIFG
import SRU
import SRU
import SRU_0
import SRU_1
import SRU_2
import CSCRU_2
import GRU
import BiSRU
import SRU_1_GPU
from TCN import TCN


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_channel1, output_channel2,
                 output_channel3):
        super(LSTMModel, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=input_size,
        #               out_channels=output_channel1,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(
        #     #     kernel_size=2,
        #     #     stride=2
        #     # )
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=output_channel1,
        #               out_channels=output_channel2,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(
        #     #     kernel_size=2,
        #     #     stride=2
        #     # )
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=output_channel2,
        #               out_channels=output_channel3,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(
        #     #     kernel_size=2,
        #     #     stride=2
        #     # )
        # )
        # self.seATT = SelfAttention_step(1, 64)
        # self.ATT = AttentionMechanism(input_size=1, hidden_size=64, sequence_length=64)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.tcn = TCN(12, 1, [128, 128, 128, 128], 2, 0.2)
        self.fc = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.2)
        # self.sru = sru.SRU(input_size, hidden_size, num_layers)
        # self.lstm = LSTM_5.Lstm(input_size, hidden_size)
        # self.sru = LSTM_0.Lstm(input_size, 128)
        self.ATT = Attention_cst.Attention(10, input_size, 64)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)

        # x = self.seATT(x)
        # x, _ = self.lstm(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape, '*' * 10)
        # x = self.ATT(x)
        # print(x.shape, '*' * 150)
        # print(x.shape)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        out_lstm, _ = self.lstm(x)
        # print(out_lstm.shape)
        out_tcn = self.tcn(x)
        # print(out_tcn.shape, '*' * 10)
        out = out_lstm + out_tcn
        out = self.dropout(out)
        out = self.fc(out)
        # print(out.shape)
        # print(out.shape)
        # out = self.fc(out)
        # print(out.shape, '*' * 5)
        return out[:, -1:, :]
