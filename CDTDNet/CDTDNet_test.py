import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
from thop import profile
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from evaluate import mean_absolute_error, symmetric_mean_absolute_percentage_error, \
    mean_squared_error, root_mean_squared_error, coefficient_of_determination, \
    NormalizedAbsoluteError, index_of_agreement
# from LSTM_Model import LSTMModel
import CDTDNet
from sklearn import metrics
# from TCN import TCN

from data_loader.data_loader_AirQuality import data_loader, denormalize

# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize

# from data_loader.data_loader_Energy import data_loader, denormalize

# from data_loader.data_loader_traffic_wavelet import data_loader, denormalize

# torch.manual_seed(seed=170)

plt.rcParams['font.sans-serif'] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

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
model.load_state_dict(torch.load(lag_path))

num_params_seq = sum(p.numel() for p in model_seq.parameters())
num_params_lag = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {num_params_seq + num_params_lag}")

criterion = nn.L1Loss()

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
    start_time = time.time()
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
    end_time = time.time()
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

print(
    f'Test Loss: {test_loss:.4f}, Test RMSE:{test_rmse:.4f}, Test RMSE_seq:{test_rmse_seq:.4f}, Test RMSE_lag:{test_rmse_lag:.4f}, Test MSE:{test_mse:.4f},'
    f'Test MAE:{test_mae:.4f}, Test SMAPE:{test_smape:.4f},'
    f'Test NAE:{test_nae:.4f},Test R2:{test_r2:.4f},Test IA:{test_ia:.4f}')

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

# rmse_values = np.sqrt(np.mean((test_predictions - true_labels) ** 2, axis=1))
#
# sorted_indices = np.argsort(rmse_values)
# sorted_rmse_values = rmse_values[sorted_indices]
#
# for i, rmse in enumerate(sorted_rmse_values):
#     sample_number = sorted_indices[i] + 1
#     print(f"sample{sample_number}: RMSE = {rmse:.6f}")
#
# file_name = "C:\\Users\\HE CONG BING\\Desktop\\data.csv"
# np.savetxt(file_name, test_predictions, delimiter=',', fmt='%.8f')
#
# model.eval()
# model_seq.eval()
# Flops = []
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         flops_seq, _ = profile(model_seq, inputs=(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :]))
#
#         outputs_seq = model_seq(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
#
#         outputs_seq = outputs_seq.unsqueeze(dim=1).unsqueeze(dim=2)
#
#         # outputs = model(outputs_seq)
#         flops_lag, _ = profile(model, inputs=(outputs_seq,))
#         flops = flops_seq + flops_lag
#         Flops.append(flops)
#
# Flops_all = sum(Flops)
# print(f'Flops:{Flops_all:.4f}')


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
