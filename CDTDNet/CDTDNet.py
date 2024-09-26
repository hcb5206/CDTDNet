import torch.nn as nn
import torch
import encoder_TCN
import decoder_TCN
import TCN_seq
import torch.nn.functional as F
import Attention_cst
import BiCSCRU
from Bi_SRU_lag_f import Bi_SRU_lag
from Bi_GRU_Lag import Bi_GRU
import CSCRU
import SRU


class LagRepair_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LagRepair_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.lstm = LSTM_11.BidirectionalLSTM(input_size, hidden_size)
        # self.lstm = Bi_SRU_lag.BidirectionalSRU(input_size, hidden_size)
        # self.lstm = Bi_GRU.BidirectionalSRU(input_size, hidden_size)
        self.fc0 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x_lag_out, _ = self.lstm(x)
        output = self.fc0(x_lag_out)
        return output


class LagRepair_CNN(nn.Module):
    def __init__(self, input_size, output_channel1, hidden_de, output_size):
        super(LagRepair_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=output_channel1,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=output_channel1,
        #               out_channels=output_channel2,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=output_channel2,
        #               out_channels=output_channel3,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1
        #               ),
        #     nn.ReLU(),
        # )

        self.lstm = nn.LSTM(output_channel1, hidden_de, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_de * 2, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        output = self.fc(x)
        return output


class TCN_CNN(nn.Module):
    def __init__(self, input_size, hidden_channel, output_size):
        super(TCN_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=hidden_channel,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channel,
                      out_channels=hidden_channel,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channel,
                      out_channels=output_size,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VAE, self).__init__()

        # self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
        #                        bidirectional=True)
        #
        # self.decoder = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim, num_layers=num_layers,
        #                        batch_first=True,
        #                        bidirectional=True)

        self.encoder = BiCSCRU.BidirectionalLSTM(input_size=input_dim, hidden_size=hidden_dim)

        self.decoder = BiCSCRU.BidirectionalLSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim)

        self.fc_output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        en_out = self.encoder(x)
        de_out = self.decoder(en_out)
        x_hat = self.fc_output(de_out)

        return x_hat


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, input_tcn, npred, flavor, seq_len, num_channels, kernel_size, dropout,
                 hidden_att_f, batch_size):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.npred = npred
        self.flavor = flavor
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.enc = encoder_TCN.Encoder(input_size, hidden_size, flavor, batch_size)
        self.dec = decoder_TCN.Decoder(hidden_size, flavor, npred, seq_len, batch_size)
        self.tcn = TCN_seq.TCN(input_size=input_tcn, num_channels=num_channels, kernel_size=kernel_size,
                               dropout=dropout)
        # self.cnn = TCN_CNN(input_size=input_tcn, hidden_channel=hidden_channel, output_size=input_tcn)
        self.start_token = torch.zeros(self.batch_size, 1, 1)
        self.ATT = Attention_cst.Attention(seq_len=seq_len, input_size=input_tcn, hidden_size=hidden_att_f)

    def forward(self, x, x_tcn):
        if self.flavor == 'lstm':
            x_att = self.ATT(x_tcn)
            x_tcn = self.tcn(x_att)
            x = torch.cat((x, x_tcn), dim=1)
            hprev, sprev, hprev_att = self.enc(x)
            ouput = self.dec(self.start_token, sprev, hprev, hprev, hprev_att)
            return ouput
        elif self.flavor == 'gru':
            x_att = self.ATT(x_tcn)
            x_tcn = self.tcn(x_att)
            x = torch.cat((x, x_tcn), dim=1)
            hprev, hprev_att = self.enc(x)
            output = self.dec(self.start_token, hprev, hprev, hprev, hprev_att)
            return output
        elif self.flavor == 'sru':
            x_att = self.ATT(x_tcn)
            x_tcn = self.tcn(x_att)
            x = torch.cat((x, x_tcn), dim=1)
            hprev, sprev, hprev_att = self.enc(x)
            ouput = self.dec(self.start_token, sprev, hprev, hprev, hprev_att)
            return ouput
        elif self.flavor == 'CSCRU':
            x_att = self.ATT(x_tcn)
            x_tcn = self.tcn(x_att)
            x = torch.cat((x, x_tcn), dim=1)
            hprev, sprev, fprev, hprev_att = self.enc(x)
            ouput = self.dec(hprev, sprev, hprev, fprev, hprev_att)
            return ouput
        elif self.flavor == '2_CSCRU':
            x_att = self.ATT(x_tcn)
            x_tcn = self.tcn(x_att)
            x = torch.cat((x, x_tcn), dim=1)
            hprev, sprev, fprev, hprev_att = self.enc(x)
            output = self.dec(hprev, sprev, hprev, fprev, hprev_att)
            return output
        elif self.flavor == 'BiCSCRU':
            x_att = self.ATT(x_tcn)
            x_tcn = self.tcn(x_att)
            # x_cnn = self.cnn(x_tcn)
            x = torch.cat((x, x_tcn), dim=1)
            hprev, sprev, fprev, hprev_att = self.enc(x)
            ouput = self.dec(hprev, sprev, hprev, fprev, hprev_att)
            return ouput
