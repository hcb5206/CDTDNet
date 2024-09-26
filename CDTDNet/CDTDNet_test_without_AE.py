import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from evaluate import mean_absolute_error, symmetric_mean_absolute_percentage_error, \
    mean_squared_error, root_mean_squared_error, coefficient_of_determination, \
    NormalizedAbsoluteError, index_of_agreement
# from LSTM_Model import LSTMModel
import CDTDNet
from sklearn import metrics
# from TCN import TCN
# from data_loader_s import train_loader, eval_loader, test_loader, denormalize
# from data_loader_s_fac import data_loader, denormalize
from data_loader.data_loader_AirQuality import data_loader, denormalize
# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize
# from data_loader.data_loader_Energy import data_loader, denormalize
# from data_loader.data_loader_traffic_wavelet import data_loader, denormalize

# torch.manual_seed(seed=381)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

model_path = 'models/Air/Air_SEQ_LSTM_1S'


# lag_path = 'models/Energy/Energy_lag_3S'


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

train_loader, eval_loader, test_loader, targets_min, targets_max = data_loader(seq_len, targets_len)
# train_loader, eval_loader, test_loader, targets_min, targets_max, train_sequences, eval_sequences, test_sequences, \
#     test_true = data_loader(seq_len, targets_len)
num_channels = cul_layers(seq_len, b=2, k=kernel_size, hidden_size=hidden_tcn, out_size=input_tcn)
# model.to(device)

model = CDTDNet.EncoderDecoder(input_size=input_size, hidden_size=hidden_size, input_tcn=input_tcn,
                                   npred=targets_len, flavor=flavor, seq_len=seq_len, num_channels=num_channels,
                                   kernel_size=kernel_size, dropout=dropout, hidden_att_f=hidden_att_f,
                                   batch_size=batch_size)
model.load_state_dict(torch.load(model_path))

num_params_lag = sum(p.numel() for p in model.parameters())
print(f"模型参数数量: {num_params_lag}")

criterion = nn.L1Loss()

model.eval()
test_loss = 0.0
test_rmse = 0.0
test_mse = 0.0
test_mae = 0.0
test_smape = 0.0
test_r2 = 0.0
test_nae = 0.0
test_ia = 0.0
test_predictions = []
true_labels = []
# output_all_s = []
with torch.no_grad():
    start_time = time.time()
    for inputs, targets in test_loader:
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        outputs = model(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
        # print(inputs.shape)
        # print(targets.shape)
        # output_all = []
        # for i in range(targets_len):
        #     outputs = model(inputs, inputs)
        #     # print(outputs.shape)
        #     output_all.append(outputs.tolist())
        #     inputs = inputs[:, :, 1:]
        #     inputs = torch.cat((inputs, outputs.view(64, 1, 1)), dim=2)

        # output_all = np.array(output_all).T
        # print(output_all.shape)
        # true_labels.extend(targets.squeeze().tolist())
        # output_all_s.extend(output_all.squeeze().tolist())
        # print(len(output_all_s))

        # outputs = model(inputs, inputs)
        # print(outputs.shape)
        # outputs = outputs.mean(dim=1)
        # targets = targets.mean(dim=1)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        outputs = denormalize(outputs, targets_min, targets_max)
        targets = denormalize(targets, targets_min, targets_max)
        rmse = root_mean_squared_error(targets.squeeze(), outputs.squeeze())
        mse = mean_squared_error(targets.squeeze(), outputs.squeeze())
        mae = mean_absolute_error(targets.squeeze(), outputs.squeeze())
        smape = symmetric_mean_absolute_percentage_error(targets.squeeze(), outputs.squeeze())
        r2 = coefficient_of_determination(targets.squeeze(), outputs.squeeze())
        nae = NormalizedAbsoluteError(targets.squeeze(), outputs.squeeze())
        ia = index_of_agreement(targets.squeeze(), outputs.squeeze())
        test_loss += loss.item()
        test_rmse += rmse
        test_mse += mse
        test_mae += mae
        test_smape += smape
        test_r2 += r2
        test_nae += nae
        test_ia += ia
        # outputs = outputs[:, -1]
        # targets = targets[:, -1]
        # print(outputs.shape, targets.shape)
        test_predictions.extend(outputs.squeeze().tolist())
        true_labels.extend(targets.squeeze().tolist())
        # output_all_s.append(output_all.squeeze().tolist())
    end_time = time.time()
test_loss /= len(test_loader)
test_rmse /= len(test_loader)
test_mse /= len(test_loader)
test_mae /= len(test_loader)
test_smape /= len(test_loader)
test_r2 /= len(test_loader)
test_nae /= len(test_loader)
test_ia /= len(test_loader)

# test_predictions = np.array(test_predictions)
# true_labels = np.array(true_labels)
# print(test_predictions.shape)
# print(true_labels.shape)

print(
    f'Test Loss: {test_loss:.4f}, Test RMSE:{test_rmse:.4f}, Test MSE:{test_mse:.4f}, '
    f'Test MAE:{test_mae:.4f}, Test SMAPE:{test_smape:.4f}, '
    f'Test NAE:{test_nae:.4f}, Test R2:{test_r2:.4f}, Test IA:{test_ia:.4f}')

print(f'Time:{end_time - start_time:.4f}')


plt.figure(figsize=(10, 5))
plt.plot(true_labels, label='Real PM2.5 concentration', color='red', linewidth=2)
plt.plot(test_predictions, label='Predict PM2.5 concentration', linestyle='--', color='blue', linewidth=2)
plt.xlabel('Time (h)', fontsize=20)
plt.ylabel('PM2.5 concentration (1023 sites)', fontsize=20)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# def save_to_csv(true_labels, test_predictions, file_path="C:\\Users\\HE CONG BING\\Desktop\\data.csv"):
#     residuals = [true - pred for true, pred in zip(true_labels, test_predictions)]
#
#     result_lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
#     print(result_lb)
#     test_statistic = result_lb['lb_stat']
#     p_values = result_lb['lb_pvalue']
#
#     print(f'Ljung-Box Test Statistic: {test_statistic}')
#     print('p-values:')
#     for lag, (lb_stat, p_value) in enumerate(zip(test_statistic, p_values)):
#         print(f'   Lag {lag + 1}: lb_stat = {lb_stat}, p-value = {p_value}')
#
#     alpha = 0.05
#     rejected = p_values < alpha
#     if any(rejected):
#         print('The null hypothesis is rejected, indicating that autocorrelation exists in the residual sequence.')
#     else:
#         print('The null hypothesis cannot be rejected, indicating that the residual sequence is white noise with no
#         autocorrelation.')
#
#     data = pd.DataFrame({
#         'CDTDNet_main Real-world data': true_labels,
#         'CDTDNet_main Predicted data': test_predictions,
#         'CDTDNet_main Residual value': residuals
#     })
#
#     data.to_csv(file_path, index=False, encoding='GBK')
#
#     print(f"The data has been successfully saved to the file '{file_path}'.")
#
#
# save_to_csv(true_labels, test_predictions)
# print(true_labels)
