import torch
import torch.nn as nn
import LSTM_decoder
import GRU_decoder
import SRU_decoder
import CSCRU_decoder_1
import CSCRU_2layers_decoder


class Decoder(nn.Module):

    def __init__(self, hidden_size, flavor, npred, seq_len, batch_szie):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 1
        self.flavor = flavor
        self.npred = npred
        self.seq_len = seq_len
        self.batch_size = batch_szie

        if flavor == 'lstm':
            self.rnn = LSTM_decoder.Lstm(self.input_size, hidden_size, npred, seq_len, batch_szie)
        elif flavor == 'gru':
            self.rnn = GRU_decoder.Lstm(self.input_size, hidden_size, npred, seq_len, batch_szie)
        elif flavor == 'sru':
            self.rnn = SRU_decoder.Lstm(self.input_size, hidden_size, npred, seq_len, batch_szie)
        elif flavor == 'CSCRU':
            self.rnn = CSCRU_decoder_1.Lstm(self.input_size, hidden_size, npred, seq_len, batch_szie)
        elif flavor == '2_CSCRU':
            self.rnn = CSCRU_2layers_decoder.Lstm(self.input_size, hidden_size, npred, seq_len, batch_szie)
            # self.rnn = CSCRU_decoder_1.Lstm(self.input_size, hidden_size, npred)
        elif flavor == 'BiCSCRU':
            self.rnn = CSCRU_2layers_decoder.Lstm(self.input_size, hidden_size, npred, seq_len, batch_szie)
            # self.rnn = CSCRU_decoder_1.Lstm(self.input_size, hidden_size, npred)

    def forward(self, x, spre, hpre, fpre, hpre_att):
        if self.flavor == 'lstm':
            output = self.rnn(x, spre, hpre, hpre_att)
            return output
        elif self.flavor == 'gru':
            output = self.rnn(x, hpre, hpre_att)
            return output
        elif self.flavor == 'sru':
            output = self.rnn(x, spre, hpre_att)
            return output
        elif self.flavor == 'CSCRU':
            output = self.rnn(x, spre, fpre, hpre_att)
            return output
        elif self.flavor == '2_CSCRU':
            output = self.rnn(x, spre, fpre, hpre_att)
            return output
        elif self.flavor == 'BiCSCRU':
            output = self.rnn(x, spre, fpre, hpre_att)
            return output
