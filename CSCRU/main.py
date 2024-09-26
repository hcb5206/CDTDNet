import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import time
from evaluate import mean_absolute_error, symmetric_mean_absolute_percentage_error, \
    mean_squared_error, root_mean_squared_error, coefficient_of_determination, \
    NormalizedAbsoluteError, index_of_agreement
from LSTM_Model import LSTMModel
from TCN import TCN
from data_loader_f import train_loader, eval_loader, test_loader, denormalize

# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# model_path = 'D:\\network_model\\model_save\\LSTM_6_time'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# torch.manual_seed(10)

input_size = 14  # Air:14, NFLX: 15
hidden_size = 64
num_layers = 1
output_size = 1
output_channel1 = 32
output_channel2 = 64
output_channel3 = 64
seq_len = 10
targets_len = 1

# train_loader, eval_loader, test_loader, targets_min, targets_max = data_loader(seq_len, targets_len)

model = LSTMModel(input_size, hidden_size, num_layers, output_size, output_channel1, output_channel2,
                  output_channel3)

# batch_size = 64
# input_size = 23
# seq_length = 10
# # num_channels = [281,477,302,558,367]
# num_channels = [64, 64, 64]
# kernel_size = 3
# dropout = 0.2
# model = TCN(input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, output_size=1)

# model.to(device)

num_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=0.001)
for name, parm in model.named_parameters():
    print(name, parm)


# class CILLoss(nn.Module):
#     def __init__(self, fs, smooth):
#         super(CILLoss, self).__init__()
#         self.fs = fs
#         self.smooth = smooth
#         self.mse = nn.MSELoss()
#         self.mae = nn.L1Loss()
#
#     def forward(self, predictions, targets):
#         # diff_2 = torch.sum((torch.diff(torch.diff(targets))) ** 2)
#         loss = self.fs * self.mse(predictions, targets) + (1 - self.fs) * self.mae(predictions, targets)
#
#         return loss

def CILoss(fs, smooth, mse, mae, predictions, targets):
    diff_2 = torch.sum((torch.diff(torch.diff(targets))) ** 2)
    loss = fs * mse(predictions, targets) + (1 - fs) * mae(predictions, targets) + smooth * diff_2

    return loss


criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = CILLoss(fs=0.05, smooth=0.1)

train_losses = []
eval_losses = []
epoch_times = []
grad_values = []
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
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        # loss = CILoss(fs, smooth, mse_l, mae_l, outputs.squeeze(), targets)
        outputs = denormalize(outputs)
        targets = denormalize(targets)
        # outputs = denormalize(outputs, targets_min, targets_max)
        # targets = denormalize(targets, targets_min, targets_max)
        targets = targets.squeeze()
        rmse = root_mean_squared_error(targets, outputs.squeeze())
        mse = mean_squared_error(targets, outputs.squeeze())
        mae = mean_absolute_error(targets, outputs.squeeze())
        smape = symmetric_mean_absolute_percentage_error(targets, outputs.squeeze())
        r2 = coefficient_of_determination(targets, outputs.squeeze())
        nae = NormalizedAbsoluteError(targets, outputs.squeeze())
        ia = index_of_agreement(targets, outputs.squeeze())
        loss.backward()

        # grad_values.append(model.lstm.blocks[0].layers[0].layers[0].wo.grad.norm().item())

        optimizer.step()
        epoch_train_loss += loss.item()
        train_rmse += rmse
        train_mse += mse
        trian_mae += mae
        train_smape += smape
        train_r2 += r2
        train_nae += nae
        train_ia += ia
    epoch_train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    train_mse /= len(train_loader)
    trian_mae /= len(train_loader)
    train_smape /= len(train_loader)
    train_r2 /= len(train_loader)
    train_nae /= len(train_loader)
    train_ia /= len(train_loader)
    train_losses.append(epoch_train_loss)
    model.eval()
    epoch_eval_loss = 0.0
    eval_rmse = 0.0
    eval_mse = 0.0
    eval_mae = 0.0
    eval_smape = 0.0
    eval_r2 = 0.0
    eval_nae = 0.0
    eval_ia = 0.0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            # loss = CILoss(fs, smooth, mse_l, mae_l, outputs.squeeze(), targets)
            outputs = denormalize(outputs)
            targets = denormalize(targets)
            # outputs = denormalize(outputs, targets_min, targets_max)
            # targets = denormalize(targets, targets_min, targets_max)
            targets = targets.squeeze()
            rmse = root_mean_squared_error(targets, outputs.squeeze())
            mse = mean_squared_error(targets, outputs.squeeze())
            mae = mean_absolute_error(targets, outputs.squeeze())
            smape = symmetric_mean_absolute_percentage_error(targets, outputs.squeeze())
            r2 = coefficient_of_determination(targets, outputs.squeeze())
            nae = NormalizedAbsoluteError(targets, outputs.squeeze())
            ia = index_of_agreement(targets, outputs.squeeze())
            epoch_eval_loss += loss.item()
            eval_rmse += rmse
            eval_mse += mse
            eval_mae += mae
            eval_smape += smape
            eval_r2 += r2
            eval_nae += nae
            eval_ia += ia
    epoch_eval_loss /= len(eval_loader)
    eval_rmse /= len(eval_loader)
    eval_mse /= len(eval_loader)
    eval_mae /= len(eval_loader)
    eval_smape /= len(eval_loader)
    eval_r2 /= len(eval_loader)
    eval_nae /= len(eval_loader)
    eval_ia /= len(eval_loader)
    eval_losses.append(epoch_eval_loss)
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {epoch_train_loss:.4f}, Eval Loss: {epoch_eval_loss:.4f},'
        f' Train RMSE: {train_rmse:.4f}, Train MSE:{train_mse:.4f},Train MAE:{trian_mae:.4f},'
        f'Train SMAPE:{train_smape:.4f},Train NAE:{train_nae:.4f},Train R2:{train_r2:.4f},Train IA:{train_ia:.4f},'
        f'Eval RMSE:{eval_rmse:.4f}, Eval MSE:{eval_mse:.4f},Eval MAE:{eval_mae:.4f},'
        f'Eval SMAPE:{eval_smape:.4f}, Eval NAE:{eval_nae:.4f},Eval R2:{eval_r2:.4f},Eval IA:{eval_ia:.4f}')
total_training_time = sum(epoch_times)
print(f'Total training time: {total_training_time:.2f}s')
# torch.save(model.state_dict(), model_path)
# print('Model saved successfully!')

plt.plot(grad_values, label='Gradient values')
plt.xlabel('Iteration')
plt.ylabel('Gradient Value')
plt.title('Gradient Changes Over Training Of The SUR-LSTM')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
plt.xlabel('Epochs')
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
        outputs = model(inputs)
        # out_lag = outputs[1:]
        # outputs = torch.cat((out_lag, out_lag[-1:]))
        loss = criterion(outputs.squeeze(), targets.squeeze())
        # loss = CILoss(fs, smooth, mse_l, mae_l, outputs.squeeze(), targets)
        outputs = denormalize(outputs)
        targets = denormalize(targets)
        # outputs = denormalize(outputs, targets_min, targets_max)
        # targets = denormalize(targets, targets_min, targets_max)
        targets = targets.squeeze()
        rmse = root_mean_squared_error(targets, outputs.squeeze())
        mse = mean_squared_error(targets, outputs.squeeze())
        mae = mean_absolute_error(targets, outputs.squeeze())
        smape = symmetric_mean_absolute_percentage_error(targets, outputs.squeeze())
        r2 = coefficient_of_determination(targets, outputs.squeeze())
        nae = NormalizedAbsoluteError(targets, outputs.squeeze())
        ia = index_of_agreement(targets, outputs.squeeze())
        test_loss += loss.item()
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
