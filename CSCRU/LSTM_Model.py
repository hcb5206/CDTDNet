import torch.nn as nn
from Recurrent_Neural_Network import RNN
from Recurrent_Neural_Network import LSTM
from Recurrent_Neural_Network import Peephole_LSTM
from Recurrent_Neural_Network import CIFG
from Recurrent_Neural_Network import CSCRU
from Recurrent_Neural_Network import SRU
from Recurrent_Neural_Network import GRU
import MGU
from Recurrent_Neural_Network import T_LSTM
from Recurrent_Neural_Network import dRNN
from Recurrent_Neural_Network import SUR_LSTM
import Attention_cst


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
        #     nn.MaxPool1d(
        #         kernel_size=2,
        #         stride=2
        #     )
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=output_channel1,
        #               out_channels=output_channel2,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        #     nn.MaxPool1d(
        #         kernel_size=2,
        #         stride=2
        #     )
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=output_channel2,
        #               out_channels=output_channel3,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        #     nn.MaxPool1d(
        #         kernel_size=2,
        #         stride=2
        #     )
        # )
        # self.seATT = SelfAttention_step(1, 64)
        # self.ATT = AttentionMechanism(input_size=1, hidden_size=64, sequence_length=64)
        # self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True,
        #                     bidirectional=True)
        self.lstm = SUR_LSTM.Lstm(input_size=input_size, hidden_size=hidden_size, hidden_out_size=hidden_size)
        # self.fc = nn.Linear(hidden_size, output_size)

        # self.sru = sru.SRU(input_size, hidden_size, num_layers)
        # self.lstm = LSTM_0.Lstm(input_size, hidden_size)
        # self.sru = LSTM_0.Lstm(input_size, 128)
        # self.ATT = Attention_cst.Attention(seq_len=10, input_size=input_size, hidden_size=64)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)

        # x = self.seATT(x)
        # x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        # x = x.transpose(1, 2)
        # x = self.ATT(x)
        # print(x.shape, '*' * 150)
        # print(x.shape)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        out, _ = self.lstm(x)
        # print(out.shape)
        # print(out.shape)
        # out = self.fc(out)
        # print(out.shape, '*' * 5)
        return out[:, :, -1]
