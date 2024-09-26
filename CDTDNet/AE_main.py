import torch
import time
import os
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluate import mean_absolute_error, symmetric_mean_absolute_percentage_error, \
    mean_squared_error, root_mean_squared_error, coefficient_of_determination, \
    NormalizedAbsoluteError, index_of_agreement
import CDTDNet
from data_loader.data_loader_AirQuality import data_loader, denormalize

# from data_loader.data_loader_Energy import data_loader, denormalize
# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize
# from data_loader.data_loader_Energy import data_loader, denormalize
# from data_loader.data_loader_traffic_wavelet import data_loader, denormalize

torch.manual_seed(seed=500)

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())

model_path = 'models/Air/Air_SEQ_LSTM_1S'
lag_path = 'models/Air/Air_lag_1S'


def cul_layers(seq_len, b, k, hidden_size, out_size):
    out_l = []
    s = ((seq_len - 1) * (b - 1) / (k - 1) + 1)
    n = math.ceil(math.log(s, b))
    for i in range(n):
        if i < n - 1:
            out_l.append(hidden_size)
        else:
            out_l.append(out_size)

    return out_l


class CILLoss(nn.Module):
    def __init__(self, fs, smooth):
        super(CILLoss, self).__init__()
        self.fs = fs
        self.smooth = smooth
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def diff_2(self, x):
        diff = torch.diff(torch.diff(x))
        return diff

    def forward(self, predictions, targets):
        diff_2 = torch.sum((self.diff_2(predictions) - self.diff_2(targets)) ** 2)
        loss = self.fs * self.mse(predictions, targets) + (1 - self.fs) * self.mae(predictions,
                                                                                   targets) + self.smooth * diff_2
        return loss


batch_size = 64  # Air:64, Energy:128, NFLX:64, Traffic: 128

seq_len = 10
targets_len = 1
input_size = 23  # Air:23, Energy:29, NFLX:15, Traffic: 22
input_tcn = 17  # Air:17, Energy:23, NFLX:9, Traffic: 16
hidden_size = 32
flavor = 'BiCSCRU'
kernel_size = 3
hidden_tcn = 64
dropout = 0.5
hidden_att_f = 64

input_dim = 1
hidden_dim = 64
output_dim = 1
train_loader, eval_loader, test_loader, targets_min, targets_max = data_loader(seq_len, targets_len)
# train_loader, eval_loader, test_loader, targets_min, targets_max, train_sequences, eval_sequences, test_sequences, \
#     test_true = data_loader(seq_len, targets_len)
num_channels = cul_layers(seq_len, b=2, k=kernel_size, hidden_size=hidden_tcn, out_size=input_tcn)
# model = seq2seq_TCN.LagRepair_LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
#                                    output_size=latent_dim)
model = CDTDNet.VAE(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
# model.to(device)

model_seq = CDTDNet.EncoderDecoder(input_size=input_size, hidden_size=hidden_size, input_tcn=input_tcn,
                                   npred=targets_len, flavor=flavor, seq_len=seq_len, num_channels=num_channels,
                                   kernel_size=kernel_size, dropout=dropout, hidden_att_f=hidden_att_f,
                                   batch_size=batch_size)
model_seq.load_state_dict(torch.load(model_path))

num_epochs = 40
optimizer = optim.Adagrad(model.parameters(), lr=0.02)
# criterion = nn.MSELoss()
criterion = CILLoss(fs=0.05, smooth=0.1)

# for name, parm in model.named_parameters():
#     print(name, parm)

train_losses = []
eval_losses = []
epoch_times = []
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    model_seq.eval()
    epoch_train_loss = 0.0
    train_rmse = 0.0
    train_mse = 0.0
    trian_mae = 0.0
    for inputs, targets in train_loader:
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        targets = targets.squeeze()
        outputs_seq = model_seq(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
        # print(outputs_seq.shape)
        outputs_de = denormalize(outputs_seq, targets_min, targets_max)
        targets_de = denormalize(targets, targets_min, targets_max)
        rmse_seq = root_mean_squared_error(targets_de, outputs_de.squeeze())
        # outputs_lag = outputs_seq[1:]
        # outputs_lag = torch.cat((outputs_lag, outputs_lag[-1:]))
        # outputs_de_lag = denormalize(outputs_lag)
        # rmse_seq_lag = root_mean_squared_error(targets_de, outputs_de_lag.squeeze())

        outputs_seq = outputs_seq.unsqueeze(dim=1).unsqueeze(dim=2)
        # print(outputs_seq.shape)

        # outputs_seq = outputs_seq.unsqueeze(dim=2)

        # inputs_seq = inputs[:, :, -targets_len:].transpose(1, 2)
        # outputs_seq = torch.cat((outputs_seq, inputs_seq), dim=2)

        # print(outputs_seq.shape)
        # outputs_seq = outputs_seq.to(device)
        # targets = targets.to(device)
        # print(outputs_seq.shape)
        # print(outputs_seq.shape)
        optimizer.zero_grad()
        outputs = model(outputs_seq).squeeze()
        # print(outputs.shape)
        # print(epoch, model.parm_lag, model.parm_bais)
        # print(outputs.shape, targets.shape)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        outputs = denormalize(outputs, targets_min, targets_max)
        targets = denormalize(targets, targets_min, targets_max)
        rmse = root_mean_squared_error(targets, outputs.squeeze())
        mse = mean_squared_error(targets, outputs.squeeze())
        mae = mean_absolute_error(targets, outputs.squeeze())
        epoch_train_loss += loss.item()
        train_rmse += rmse
        train_mse += mse
        trian_mae += mae
    epoch_train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    train_mse /= len(train_loader)
    trian_mae /= len(train_loader)
    train_losses.append(epoch_train_loss)
    model.eval()
    model_seq.eval()
    epoch_eval_loss = 0.0
    eval_rmse = 0.0
    eval_mse = 0.0
    eval_mae = 0.0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            targets = targets.squeeze()
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            outputs_seq = model_seq(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
            outputs_de = denormalize(outputs_seq, targets_min, targets_max)
            targets_de = denormalize(targets_de, targets_min, targets_max)
            rmse_seq = root_mean_squared_error(targets_de, outputs_de.squeeze())
            # outputs_lag = outputs_seq[1:]
            # outputs_lag = torch.cat((outputs_lag, outputs_lag[-1:]))
            # outputs_de_lag = denormalize(outputs_lag)
            # rmse_seq_lag = root_mean_squared_error(targets_de, outputs_de_lag.squeeze())

            outputs_seq = outputs_seq.unsqueeze(dim=1).unsqueeze(dim=2)

            # outputs_seq = outputs_seq.unsqueeze(dim=2)

            # inputs_seq = inputs[:, :, -targets_len:].transpose(1, 2)
            # outputs_seq = torch.cat((outputs_seq, inputs_seq), dim=2)

            # outputs_seq = outputs_seq.to(device)
            # targets = targets.to(device)
            outputs = model(outputs_seq).squeeze()
            loss = criterion(outputs.squeeze(), targets)
            outputs = denormalize(outputs, targets_min, targets_max)
            targets = denormalize(targets, targets_min, targets_max)
            rmse = root_mean_squared_error(targets, outputs.squeeze())
            mse = mean_squared_error(targets, outputs.squeeze())
            mae = mean_absolute_error(targets, outputs.squeeze())
            epoch_eval_loss += loss.item()
            eval_rmse += rmse
            eval_mse += mse
            eval_mae += mae
    epoch_eval_loss /= len(eval_loader)
    eval_rmse /= len(eval_loader)
    eval_mse /= len(eval_loader)
    eval_mae /= len(eval_loader)
    eval_losses.append(epoch_eval_loss)
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {epoch_train_loss:.4f}, '
        f'Eval Loss: {epoch_eval_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MSE:{train_mse:.4f},'
        f'Train MAE:{trian_mae:.4f}, Eval RMSE:{eval_rmse:.4f}, Eval MSE:{eval_mse:.4f},Eval MAE:{eval_mae:.4f}')
total_training_time = sum(epoch_times)
print(f'Total training time: {total_training_time:.2f}s')
torch.save(model.state_dict(), lag_path)
print('Model saved successfully!')

model.eval()
model_seq.eval()
test_loss = 0.0
test_rmse = 0.0
test_rmse_seq = 0.0
test_rmse_lag = 0.0
test_mse = 0.0
test_mae = 0.0
test_smape = 0.0
test_r2 = 0.0
test_nae = 0.0
test_ia = 0.0
test_predictions = []
true_labels = []
with torch.no_grad():
    for inputs, targets in test_loader:
        targets = targets.squeeze()
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        # targets = targets[:-1]
        # print(targets.shape)
        outputs_seq = model_seq(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
        outputs_de = denormalize(outputs_seq, targets_min, targets_max)
        targets_de = denormalize(targets, targets_min, targets_max)
        rmse_seq = root_mean_squared_error(targets_de, outputs_de.squeeze())
        # outputs_lag = outputs_seq[1:]
        # outputs_lag = torch.cat((outputs_lag, outputs_lag[-1:]))
        # outputs_de_lag = denormalize(outputs_lag)
        # rmse_seq_lag = root_mean_squared_error(targets_de, outputs_de_lag.squeeze())
        # outputs_lag = outputs[1:]
        # outputs = torch.cat((outputs_lag, outputs_lag[-1:]))
        # print(outputs.shape)

        outputs_seq = outputs_seq.unsqueeze(dim=1).unsqueeze(dim=2)

        # outputs_seq = outputs_seq.unsqueeze(dim=2)

        # inputs_seq = inputs[:, :, -targets_len:].transpose(1, 2)
        # outputs_seq = torch.cat((outputs_seq, inputs_seq), dim=2)

        # outputs_seq = outputs_seq.to(device)
        # targets = targets.to(device)
        outputs = model(outputs_seq).squeeze()
        loss = criterion(outputs.squeeze(), targets)
        outputs = denormalize(outputs, targets_min, targets_max)
        targets = denormalize(targets, targets_min, targets_max)
        rmse = root_mean_squared_error(targets, outputs.squeeze())
        mse = mean_squared_error(targets, outputs.squeeze())
        mae = mean_absolute_error(targets, outputs.squeeze())
        smape = symmetric_mean_absolute_percentage_error(targets, outputs.squeeze())
        r2 = coefficient_of_determination(targets, outputs.squeeze())
        # r2 = metrics.r2_score(targets, outputs.squeeze())
        nae = NormalizedAbsoluteError(targets, outputs.squeeze())
        ia = index_of_agreement(targets, outputs.squeeze())
        test_loss += loss.item()
        test_rmse_seq += rmse_seq
        # test_rmse_lag += rmse_seq_lag
        test_rmse += rmse
        test_mse += mse
        test_mae += mae
        test_smape += smape
        test_r2 += r2
        test_nae += nae
        test_ia += ia
        test_predictions.extend(outputs.squeeze().tolist())
        true_labels.extend(targets.tolist())
test_loss /= len(test_loader)
test_rmse_seq /= len(test_loader)
test_rmse_lag /= len(test_loader)
test_rmse /= len(test_loader)
test_mse /= len(test_loader)
test_mae /= len(test_loader)
test_smape /= len(test_loader)
test_r2 /= len(test_loader)
test_nae /= len(test_loader)
test_ia /= len(test_loader)

print(len(true_labels), len(test_predictions))

print(
    f'Test Loss: {test_loss:.4f}, Test RMSE:{test_rmse:.4f}, Test RMSE_seq:{test_rmse_seq:.4f}, Test RMSE_lag:{test_rmse_lag:.4f}, Test MSE:{test_mse:.4f},'
    f'Test MAE:{test_mae:.4f}, Test SMAPE:{test_smape:.4f},'
    f'Test NAE:{test_nae:.4f},Test R2:{test_r2:.4f},Test IA:{test_ia:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(true_labels, label='True value', color='red', linewidth=2)
plt.plot(test_predictions, label='Predicted value', linestyle='--', color='blue', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
