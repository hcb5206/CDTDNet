import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
"""
没有验证集
"""
data = pd.read_csv("C:\\Users\\HE CONG BING\\Desktop\\1001.csv")
# data = pd.read_csv("C:\\Users\\HE CONG BING\\Desktop\\1001_r_all.csv")

PM25 = data['PM25_Concentration'].values

t = np.arange(8760)
PM25_s = pd.DataFrame({'Time': t, 'PM25_Concentration': PM25})


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


lag_order = 2
for i in range(1, lag_order + 1):
    PM25_s[f'Value_Lag_{i}'] = PM25_s['PM25_Concentration'].shift(i)

PM25_s['Value_Diff'] = PM25_s['PM25_Concentration'].diff()
PM25_s = linear_interpolate(PM25_s, 'Value_Lag_1')
PM25_s = linear_interpolate(PM25_s, 'Value_Lag_2')
PM25_s = linear_interpolate(PM25_s, 'Value_Diff')

air_imformation = [
    data['PM25_Concentration'].tolist(),
    data['PM10_Concentration'].tolist(),
    data['NO2_Concentration'].tolist(),
    data['CO_Concentration'].tolist(),
    data['O3_Concentration'].tolist(),
    data['SO2_Concentration'].tolist(),
    data['weather'].tolist(),
    data['temperature'].tolist(),
    data['pressure'].tolist(),
    data['humidity'].tolist(),
    data['wind_speed'].tolist(),
    data['wind_direction'].tolist(),
    data['time_num'].tolist(),
    data['date_num'].tolist(),
    PM25_s['Value_Lag_1'].tolist(),
    PM25_s['Value_Lag_2'].tolist(),
    PM25_s['Value_Diff'].tolist()
]


def normalize_data(data, data_all):
    data_train = []
    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        data_s = (data_all[i] - min_val) / (max_val - min_val)
        data_train.append(data_s)
        if i == 0:
            data_target = data_s
            targets_min = min_val
            targets_max = max_val
    return data_train, data_target, targets_min, targets_max


def denormalize(normalized_value):
    original_min, original_max = targets_min, targets_max
    original_value = normalized_value * (original_max - original_min) + original_min
    return original_value


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data[0]) - seq_length):
        seq = [lst[i:i + seq_length] for lst in data]
        sequences.append(seq)
    return np.array(sequences)


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


def time_imformation(data_tensor):
    mean_value_first_half = data_tensor[:, :, :5].mean(dim=2, keepdim=True)
    mean_value_second_half = data_tensor[:, :, 5:].mean(dim=2, keepdim=True)
    median_value_first_half = data_tensor[:, :, :5].median(dim=2, keepdim=True)[0]
    median_value_second_half = data_tensor[:, :, 5:].median(dim=2, keepdim=True)[0]
    max_value_first_half = data_tensor[:, :, :5].max(dim=2, keepdim=True)[0]
    max_value_second_half = data_tensor[:, :, 5:].max(dim=2, keepdim=True)[0]
    min_value_first_half = data_tensor[:, :, :5].min(dim=2, keepdim=True)[0]
    min_value_second_half = data_tensor[:, :, 5:].min(dim=2, keepdim=True)[0]
    std_value_first_half = data_tensor[:, :, :5].std(dim=2, keepdim=True)
    std_value_second_half = data_tensor[:, :, 5:].std(dim=2, keepdim=True)

    mean_filled_data_tensor = torch.cat((mean_value_first_half.expand_as(data_tensor[:, :, :5]),
                                         mean_value_second_half.expand_as(data_tensor[:, :, 5:])), dim=2)

    median_filled_data_tensor = torch.cat((median_value_first_half.expand_as(data_tensor[:, :, :5]),
                                           median_value_second_half.expand_as(data_tensor[:, :, 5:])), dim=2)

    max_filled_data_tensor = torch.cat((max_value_first_half.expand_as(data_tensor[:, :, :5]),
                                        max_value_second_half.expand_as(data_tensor[:, :, 5:])), dim=2)

    min_filled_data_tensor = torch.cat((min_value_first_half.expand_as(data_tensor[:, :, :5]),
                                        min_value_second_half.expand_as(data_tensor[:, :, 5:])), dim=2)

    std_filled_data_tensor = torch.cat((std_value_first_half.expand_as(data_tensor[:, :, :5]),
                                        std_value_second_half.expand_as(data_tensor[:, :, 5:])), dim=2)
    coef_data_tensor = create_coef(data_tensor, 1)
    concatenated_tensor = torch.cat(
        (mean_filled_data_tensor, median_filled_data_tensor, max_filled_data_tensor,
         min_filled_data_tensor, std_filled_data_tensor, coef_data_tensor), dim=1)
    return concatenated_tensor


seq_length = 10

train_split = round(len(PM25) * 0.80)
test_split = round(len(PM25) * 0.10)
# print(train_split, train_split + test_split, test_split)

train_size = train_split
eval_size = test_split
test_size = test_split
PM25_train_size = train_split + seq_length
PM25_eval_size = train_split + test_split
PM25_test_size = train_split + test_split + seq_length
PM25_size = train_split + test_split + test_split

num_data = [lst[:train_size + eval_size] for lst in air_imformation]
air_normalize, PM25_normalize, targets_min, targets_max = normalize_data(num_data, air_imformation)
# air_normalize, PM25_normalize = air_imformation, PM25

train_data = [lst[:train_size + eval_size] for lst in air_normalize]
test_data = [lst[train_size + eval_size:train_size + eval_size + test_size] for lst in air_normalize]

train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)
train_sequences_s = train_sequences[:, 0, :]
test_sequences_s = test_sequences[:, 0, :]

train_targets = PM25_normalize[seq_length:train_size + eval_size]
test_targets = PM25_normalize[PM25_test_size:PM25_size]

train_inputs = torch.Tensor(train_sequences)
test_inputs = torch.Tensor(test_sequences)
train_inputs_s = torch.Tensor(train_sequences_s).unsqueeze(dim=1)
test_inputs_s = torch.Tensor(test_sequences_s).unsqueeze(dim=1)
train_targets = torch.Tensor(train_targets)
test_targets = torch.Tensor(test_targets)
train_inputs_s = time_imformation(train_inputs_s)
test_inputs_s = time_imformation(test_inputs_s)

train_inputs = torch.cat((train_inputs, train_inputs_s), dim=1)
test_inputs = torch.cat((test_inputs, test_inputs_s), dim=1)
print(train_inputs.shape, test_inputs.shape, train_targets.shape, test_targets.shape)

train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)