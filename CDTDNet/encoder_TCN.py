import torch.nn as nn
import torch
import LSTM_encoder
import GRU_encoder
import SRU_encoder
import CSCRU_encoder
import CSCRU_2layers_encoder
import BiCSCRU_encoder


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, flavor, batch_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.flavor = flavor
        self.batch_size = batch_size
        self.fc_o = nn.Linear(hidden_size, 1)
        self.fc_o_bi = nn.Linear(hidden_size * 2, 1)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

        if flavor == 'lstm':
            self.rnn = LSTM_encoder.Lstm(input_size, hidden_size, batch_size)
        elif flavor == 'gru':
            self.rnn = GRU_encoder.Lstm(input_size, hidden_size, batch_size)
        elif flavor == 'sru':
            self.rnn = SRU_encoder.Lstm(input_size, hidden_size, batch_size)
        elif flavor == 'CSCRU':
            self.rnn = CSCRU_encoder.Lstm(input_size, hidden_size, batch_size)
        elif flavor == '2_CSCRU':
            self.rnn = CSCRU_2layers_encoder.MultiLayerLSTM(input_size, hidden_size, batch_size)
        elif flavor == 'BiCSCRU':
            self.rnn = BiCSCRU_encoder.BidirectionalLSTM(input_size, hidden_size, batch_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.flavor == 'lstm':
            hidden, cell, hidden_att = self.rnn(x)
            return hidden, cell, hidden_att
        elif self.flavor == 'gru':
            hidden, hidden_att = self.rnn(x)
            return hidden, hidden_att
        elif self.flavor == 'sru':
            hidden, cell, hidden_att = self.rnn(x)
            hidden = self.fc_o(hidden)
            return hidden, cell, hidden_att
        elif self.flavor == 'CSCRU':
            hidden, cell, fell, hidden_att = self.rnn(x)
            hidden = self.fc_o(hidden)
            return hidden, cell, fell, hidden_att
        elif self.flavor == '2_CSCRU':
            hidden, cell, fell, hidden_att = self.rnn(x)
            hidden = self.fc_o(hidden)
            return hidden, cell, fell, hidden_att
        elif self.flavor == 'BiCSCRU':
            hidden, cell_forward, fell_forward, cell_backward, fell_backward, hidden_forward, hidden_backward = \
                self.rnn(x)
            hidden = self.fc_o_bi(hidden)
            cell = torch.cat((cell_forward, cell_backward), dim=2)
            fell = torch.cat((fell_forward, fell_backward), dim=2)
            hidden_att = torch.cat((hidden_forward, hidden_backward), dim=2)
            cell = self.fc_c(cell)
            fell = self.fc_c(fell)
            hidden_att = self.fc_c(hidden_att)
            return hidden, cell, fell, hidden_att
