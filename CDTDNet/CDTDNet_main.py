import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import time
from evaluate import mean_absolute_error, symmetric_mean_absolute_percentage_error, \
    mean_squared_error, root_mean_squared_error, coefficient_of_determination, \
    NormalizedAbsoluteError, index_of_agreement
import CDTDNet
import math
from Loss_Function import CILLoss

from data_loader.data_loader_AirQuality import data_loader, denormalize
# from data_loader.data_loader_AirQuality_Gaussian import data_loader, denormalize
# from data_loader.data_loader_Energy import data_loader, denormalize
# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize
# from data_loader.data_loader_traffic_wavelet import data_loader, denormalize

torch.manual_seed(seed=381)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# model_path = 'models/NFLX/NFLX_SEQ_LSTM_1S'


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())


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
# noise_std = 0.5

train_loader, eval_loader, test_loader, targets_min, targets_max = data_loader(seq_len, targets_len)
num_channels = cul_layers(seq_len, b=2, k=kernel_size, hidden_size=hidden_tcn, out_size=input_tcn)
print(num_channels, targets_min, targets_max)
model = CDTDNet.EncoderDecoder(input_size=input_size, hidden_size=hidden_size, input_tcn=input_tcn,
                               npred=targets_len, flavor=flavor, seq_len=seq_len, num_channels=num_channels,
                               kernel_size=kernel_size, dropout=dropout, hidden_att_f=hidden_att_f,
                               batch_size=batch_size)
# model.to(device)

num_epochs = 150

optimizer_ende = optim.Adam(model.parameters(), lr=0.001)

loss = 'MAE'  # MSE, MAE, CIL


# optimizer_lag = optim.Adam(model.lag_r.parameters(), lr=0.001)
# for name, parm in model.named_parameters():
#     print(name, parm)


def clip_gradients(model, clip_value):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)


criterion = None
if loss == 'MSE':
    criterion = nn.MSELoss()
elif loss == 'MAE':
    criterion = nn.L1Loss()
elif loss == 'CIL':
    criterion = CILLoss(fs=0.08, smooth=0.2)

train_losses = []
eval_losses = []
epoch_times = []
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    train_rmse = 0.0
    train_mse = 0.0
    trian_mae = 0.0
    train_smape = 0.0
    train_r2 = 0.0
    train_nae = 0.0
    train_ia = 0.0
    for inputs, targets in train_loader:
        optimizer_ende.zero_grad()
        outputs = model(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer_ende.step()
        outputs = denormalize(outputs, targets_min, targets_max)
        targets = denormalize(targets, targets_min, targets_max)
        rmse = root_mean_squared_error(targets.squeeze(), outputs.squeeze())
        mse = mean_squared_error(targets.squeeze(), outputs.squeeze())
        mae = mean_absolute_error(targets.squeeze(), outputs.squeeze())
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
    epoch_eval_loss = 0.0
    eval_rmse = 0.0
    eval_mse = 0.0
    eval_mae = 0.0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            outputs = model(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
            loss = criterion(outputs.squeeze(), targets.squeeze())
            outputs = denormalize(outputs, targets_min, targets_max)
            targets = denormalize(targets, targets_min, targets_max)
            rmse = root_mean_squared_error(targets.squeeze(), outputs.squeeze())
            mse = mean_squared_error(targets.squeeze(), outputs.squeeze())
            mae = mean_absolute_error(targets.squeeze(), outputs.squeeze())
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
        f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {epoch_train_loss:.4f}, Eval Loss: {epoch_eval_loss:.4f},'
        f' Train RMSE: {train_rmse:.4f}, Train MSE:{train_mse:.4f}, Train MAE:{trian_mae:.4f}, '
        f'Eval RMSE:{eval_rmse:.4f}, Eval MSE:{eval_mse:.4f}, Eval MAE:{eval_mae:.4f}')

total_training_time = sum(epoch_times)
print(f'Total training time: {total_training_time:.2f}s')
# torch.save(model.state_dict(), model_path)
# print('Model saved successfully!')

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

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
with torch.no_grad():
    for inputs, targets in test_loader:
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        outputs = model(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
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
        test_predictions.extend(outputs.squeeze().tolist())
        true_labels.extend(targets.squeeze().tolist())
test_loss /= len(test_loader)
test_rmse /= len(test_loader)
test_mse /= len(test_loader)
test_mae /= len(test_loader)
test_smape /= len(test_loader)
test_r2 /= len(test_loader)
test_nae /= len(test_loader)
test_ia /= len(test_loader)

print(
    f'Test Loss: {test_loss:.4f}, Test RMSE:{test_rmse:.4f}, Test MSE:{test_mse:.4f}, '
    f'Test MAE:{test_mae:.4f}, Test SMAPE:{test_smape:.4f}, '
    f'Test NAE:{test_nae:.4f}, Test R2:{test_r2:.4f}, Test IA:{test_ia:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(true_labels, label='True value', color='red', linewidth=2)
plt.plot(test_predictions, label='Predicted value', linestyle='--', color='blue', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
