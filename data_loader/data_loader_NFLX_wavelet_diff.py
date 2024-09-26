import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import pywt
import warnings
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore",
                        message="Level value of .* is too high: all coefficients will experience boundary effects.",
                        category=UserWarning)


def dm_test(data):
    result = adfuller(data)

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')

    if p_value <= 0.05:
        print('拒绝原假设，时间序列是平稳的。')
    else:
        print('无法拒绝原假设，时间序列可能是非平稳的。')


def linear_interpolate(df, column):
    missing_indices = df[df[column].isnull()].index
    for index in missing_indices:
        prev_index = df[column].last_valid_index()
        next_index = df[column].first_valid_index()
        if prev_index is not None and next_index is not None:
            prev_value = df.at[prev_index, column]
            next_value = df.at[next_index, column]
            interpolated_value = (prev_value + next_value) / 2
            df.at[index, column] = interpolated_value

    return df


# def normalize_data(data, data_all):
#     data_train = []
#     for i in range(len(data)):
#         min_val = np.min(data[i])
#         max_val = np.max(data[i])
#         data_s = (data_all[i] - min_val) / (max_val - min_val)
#         data_train.append(data_s)
#         if i == 0:
#             data_target = data_s
#             targets_min = min_val
#             targets_max = max_val
#     return data_train, data_target, targets_min, targets_max

def normalize_data(data_train, data_eval, data_test, train_targets, eval_targets, test_targets):
    data_c = np.concatenate((data_train, data_eval), axis=0)
    data_targets = np.concatenate((train_targets, eval_targets), axis=0)
    data_train_n = np.zeros_like(data_train)
    data_eval_n = np.zeros_like(data_eval)
    data_test_n = np.zeros_like(data_test)
    train_targets_n = np.zeros_like(train_targets)
    eval_targets_n = np.zeros_like(eval_targets)
    test_targets_n = np.zeros_like(test_targets)
    for i in range(data_c.shape[1]):
        min_val = np.min(data_c[:, i, :])
        max_val = np.max(data_c[:, i, :])
        data_train_n[:, i, :] = (data_train[:, i, :] - min_val) / (max_val - min_val)
        data_eval_n[:, i, :] = (data_eval[:, i, :] - min_val) / (max_val - min_val)
        data_test_n[:, i, :] = (data_test[:, i, :] - min_val) / (max_val - min_val)
        if i == 0:
            min_t = np.min(data_targets)
            max_t = np.max(data_targets)
            train_targets_n = (train_targets - min_t) / (max_t - min_t)
            eval_targets_n = (eval_targets - min_t) / (max_t - min_t)
            test_targets_n = (test_targets - min_t) / (max_t - min_t)
            targets_min = min_t
            targets_max = max_t

    return data_train_n, data_eval_n, data_test_n, train_targets_n, eval_targets_n, test_targets_n, targets_min, targets_max


def denormalize(normalized_value, targets_min, targets_max):
    original_min, original_max = targets_min, targets_max
    original_value = normalized_value * (original_max - original_min) + original_min
    return original_value


# def create_sequences(data, seq_length):
#     sequences = []
#     for i in range(len(data[0]) - seq_length):
#         seq = [lst[i:i + seq_length] for lst in data]
#         sequences.append(seq)
#     return np.array(sequences)

def create_sequences(data, seq_length, target_length):
    sequences = []
    for i in range(len(data[0]) - seq_length - target_length + 1):
        seq = [lst[i:i + seq_length] for lst in data]
        sequences.append(seq)
    return np.array(sequences)


def create_target_sequences(data, target_length):
    sequences = []
    for i in range(len(data) - target_length + 1):
        seq = data[i:i + target_length]
        sequences.append(seq)
    return np.array(sequences)


def wavelet_smoothing(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = (pywt.threshold(coeff, 0.1, mode="soft") for coeff in coeffs[1:])
    smoothed_data = pywt.waverec(coeffs, wavelet)
    if data.shape[0] % 2 == 0:
        smoothed_data = smoothed_data
    else:
        smoothed_data = smoothed_data[:-1]
    return smoothed_data


# def wavelet_smoothing_s(data, wavelet='db4', level=3):
#     coeffs = pywt.wavedec(data, wavelet, level=level)
#     coeffs[1:] = (pywt.threshold(coeff, 0.1, mode="soft") for coeff in coeffs[1:])
#     smoothed_data = pywt.waverec(coeffs, wavelet)
#     return smoothed_data


def wavelet_pro(data):
    smooth_data = np.zeros_like(data)
    for sample_idx in range(data.shape[0]):
        for feature_idx in range(data.shape[1]):
            smooth_data[sample_idx, feature_idx, :] = wavelet_smoothing(data[sample_idx, feature_idx, :])

    return smooth_data


def create_coef(data, lag):
    data_output_np = []
    data = data.squeeze()
    data_np = data.numpy()
    for i in range(data_np.shape[0]):
        std1 = np.std(data_np[i, lag:])
        std2 = np.std(data_np[i, :-lag])
        if std1 > 1e-6 and std2 > 1e-6:
            coef = np.corrcoef(data_np[i, lag:], data_np[i, :-lag])[0, 1]
        else:
            coef = 0.0
        coef_expand = np.full_like(data_np[i, :], coef)
        data_output_np.append(coef_expand)
    output_data = torch.Tensor(np.array(data_output_np)).unsqueeze(dim=1)
    return output_data


def time_imformation(data_tensor, data_half):
    mean_value_first_half = data_tensor[:, :, :data_half].mean(dim=2, keepdim=True)
    mean_value_second_half = data_tensor[:, :, data_half:].mean(dim=2, keepdim=True)
    median_value_first_half = data_tensor[:, :, :data_half].median(dim=2, keepdim=True)[0]
    median_value_second_half = data_tensor[:, :, data_half:].median(dim=2, keepdim=True)[0]
    max_value_first_half = data_tensor[:, :, :data_half].max(dim=2, keepdim=True)[0]
    max_value_second_half = data_tensor[:, :, data_half:].max(dim=2, keepdim=True)[0]
    min_value_first_half = data_tensor[:, :, :data_half].min(dim=2, keepdim=True)[0]
    min_value_second_half = data_tensor[:, :, data_half:].min(dim=2, keepdim=True)[0]
    std_value_first_half = data_tensor[:, :, :data_half].std(dim=2, keepdim=True)
    std_value_second_half = data_tensor[:, :, data_half:].std(dim=2, keepdim=True)

    mean_filled_data_tensor = torch.cat((mean_value_first_half.expand_as(data_tensor[:, :, :data_half]),
                                         mean_value_second_half.expand_as(data_tensor[:, :, data_half:])), dim=2)

    median_filled_data_tensor = torch.cat((median_value_first_half.expand_as(data_tensor[:, :, :data_half]),
                                           median_value_second_half.expand_as(data_tensor[:, :, data_half:])), dim=2)

    max_filled_data_tensor = torch.cat((max_value_first_half.expand_as(data_tensor[:, :, :data_half]),
                                        max_value_second_half.expand_as(data_tensor[:, :, data_half:])), dim=2)

    min_filled_data_tensor = torch.cat((min_value_first_half.expand_as(data_tensor[:, :, :data_half]),
                                        min_value_second_half.expand_as(data_tensor[:, :, data_half:])), dim=2)

    std_filled_data_tensor = torch.cat((std_value_first_half.expand_as(data_tensor[:, :, :data_half]),
                                        std_value_second_half.expand_as(data_tensor[:, :, data_half:])), dim=2)
    coef_data_tensor = create_coef(data_tensor, 1)
    concatenated_tensor = torch.cat(
        (mean_filled_data_tensor, median_filled_data_tensor, max_filled_data_tensor,
         min_filled_data_tensor, std_filled_data_tensor, coef_data_tensor), dim=1)
    return concatenated_tensor


def data_loader(seq_len, targets_len):
    data = pd.read_csv("..\\data\\Stocks\\NFLX.csv")

    targets = data['High'].values

    t = np.arange(len(targets))
    targets_s = pd.DataFrame({'Time': t, 'High': targets})

    lag_order = 2
    for i in range(1, lag_order + 1):
        targets_s[f'Value_Lag_{i}'] = targets_s['High'].shift(i)

    targets_s['Value_Diff'] = targets_s['High'].diff()
    # print(targets_s['High'])
    # print(targets_s['Value_Diff'])
    targets_s = linear_interpolate(targets_s, 'Value_Lag_1')
    targets_s = linear_interpolate(targets_s, 'Value_Lag_2')
    targets_s = linear_interpolate(targets_s, 'Value_Diff')
    # print(targets_s['Value_Diff'])

    all_imformation = [
        data['High'].tolist(),
        data['Open'].tolist(),
        data['Low'].tolist(),
        data['Close'].tolist(),
        data['Adj Close'].tolist(),
        data['Volume'].tolist(),
        targets_s['Value_Lag_1'].tolist(),
        targets_s['Value_Lag_2'].tolist(),
        targets_s['Value_Diff'].tolist()
    ]

    seq_length = seq_len
    seq_half = seq_length // 2

    train_split = round(len(targets) * 0.80)
    test_split = round(len(targets) * 0.10)
    # print(train_split, train_split + test_split, test_split)

    train_size = train_split
    eval_size = test_split
    test_size = test_split
    PM25_train_size = train_split + seq_length
    PM25_eval_size = train_split + test_split
    PM25_test_size = train_split + test_split + seq_length
    PM25_size = train_split + test_split + test_split

    # num_data = [lst[:train_size + eval_size] for lst in air_imformation]
    # air_normalize, PM25_normalize, targets_min, targets_max = normalize_data(num_data, air_imformation)
    air_normalize, PM25_normalize = all_imformation, targets

    train_data = [lst[:train_size] for lst in air_normalize]
    eval_data = [lst[train_size:train_size + eval_size] for lst in air_normalize]
    test_data = [lst[train_size + eval_size:train_size + eval_size + test_size] for lst in air_normalize]

    train_sequences = create_sequences(train_data, seq_length, targets_len)
    eval_sequences = create_sequences(eval_data, seq_length, targets_len)
    test_sequences = create_sequences(test_data, seq_length, targets_len)

    train_sequences_wavelet = wavelet_pro(train_sequences)
    eval_sequences_wavelet = wavelet_pro(eval_sequences)
    test_sequences_wavelet = wavelet_pro(test_sequences)

    train_targets_s = PM25_normalize[seq_length:train_size]
    eval_targets_s = PM25_normalize[PM25_train_size:PM25_eval_size]
    test_targets_s = PM25_normalize[PM25_test_size:PM25_size]

    train_targets_s = create_target_sequences(train_targets_s, targets_len)
    eval_targets_s = create_target_sequences(eval_targets_s, targets_len)
    test_targets_s = create_target_sequences(test_targets_s, targets_len)

    train_targets_s_diff = train_targets_s[:, -1] - train_sequences[:, 0, -1]
    eval_targets_s_diff = eval_targets_s[:, -1] - eval_sequences[:, 0, -1]
    test_targets_s_diff = test_targets_s[:, -1] - test_sequences[:, 0, -1]
    # dm_test(train_targets_s_diff)
    # dm_test(eval_targets_s_diff)
    # dm_test(test_targets_s_diff)
    # print(train_targets_s.shape, eval_targets_s.shape, test_targets_s. shape)

    # print(train_sequences[:, 0, -1])
    # print(train_targets_s[:, -1])
    # print(train_targets_s[:, -1] - train_sequences[:, 0, -1])

    train_out, eval_out, test_out, train_targets, eval_targets, test_targets, targets_min, targets_max = normalize_data(
        train_sequences_wavelet, eval_sequences_wavelet, test_sequences_wavelet, train_targets_s_diff,
        eval_targets_s_diff,
        test_targets_s_diff)
    train_sequences_s = train_out[:, 0, :]
    eval_sequences_s = eval_out[:, 0, :]
    test_sequences_s = test_out[:, 0, :]

    train_inputs = torch.Tensor(train_out)
    eval_inputs = torch.Tensor(eval_out)
    test_inputs = torch.Tensor(test_out)
    train_inputs_s = torch.Tensor(train_sequences_s).unsqueeze(dim=1)
    eval_inputs_s = torch.Tensor(eval_sequences_s).unsqueeze(dim=1)
    test_inputs_s = torch.Tensor(test_sequences_s).unsqueeze(dim=1)
    train_targets = torch.Tensor(train_targets)
    eval_targets = torch.Tensor(eval_targets)
    test_targets = torch.Tensor(test_targets)
    train_inputs_s = time_imformation(train_inputs_s, seq_half)
    eval_inputs_s = time_imformation(eval_inputs_s, seq_half)
    test_inputs_s = time_imformation(test_inputs_s, seq_half)

    train_inputs = torch.cat((train_inputs, train_inputs_s), dim=1)
    eval_inputs = torch.cat((eval_inputs, eval_inputs_s), dim=1)
    test_inputs = torch.cat((test_inputs, test_inputs_s), dim=1)
    print(targets_min, targets_max)
    print(train_inputs.shape, eval_inputs.shape, test_inputs.shape, train_targets.shape, eval_targets.shape,
          test_targets.shape)

    train_dataset = TensorDataset(train_inputs, train_targets)
    eval_dataset = TensorDataset(eval_inputs, eval_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_loader, eval_loader, test_loader, targets_min, targets_max, train_sequences[:, 0, -1], \
           eval_sequences[:, 0, -1], test_sequences[:, 0, -1], test_targets_s[:, -1]


if __name__ == '__main__':
    data_loader(10, 1)
