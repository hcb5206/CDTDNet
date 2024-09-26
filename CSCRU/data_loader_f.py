import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

data = pd.read_csv("..\\data\\Air Quality\\1001.csv")

PM25 = data['PM25_Concentration'].values
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
    data['date_num'].tolist()
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


# def normalize_data_target(data, data_all):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     data_s = (data_all - min_val) / (max_val - min_val)
#     return data_s


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data[0]) - seq_length):
        seq = [lst[i:i + seq_length] for lst in data]
        sequences.append(seq)
    return np.array(sequences)


seq_length = 10

train_split = round(len(PM25) * 0.80)
test_split = round(len(PM25) * 0.10)

train_size = train_split
eval_size = test_split
test_size = test_split
PM25_train_size = train_split + seq_length
PM25_eval_size = train_split + test_split
PM25_test_size = train_split + test_split + seq_length
PM25_size = train_split + test_split + test_split

num_data = [lst[:train_size + eval_size] for lst in air_imformation]
air_normalize, PM25_normalize, targets_min, targets_max = normalize_data(num_data, air_imformation)

train_data = [lst[:train_size] for lst in air_normalize]
eval_data = [lst[train_size:train_size + eval_size] for lst in air_normalize]
test_data = [lst[train_size + eval_size:train_size + eval_size + test_size] for lst in air_normalize]

train_sequences = create_sequences(train_data, seq_length)
eval_sequences = create_sequences(eval_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)
train_targets = PM25_normalize[seq_length:train_size]
eval_targets = PM25_normalize[PM25_train_size:PM25_eval_size]
test_targets = PM25_normalize[PM25_test_size:PM25_size]

train_inputs = torch.Tensor(train_sequences)
eval_inputs = torch.Tensor(eval_sequences)
test_inputs = torch.Tensor(test_sequences)
train_targets = torch.Tensor(train_targets)
eval_targets = torch.Tensor(eval_targets)
test_targets = torch.Tensor(test_targets)
# print(train_inputs.shape)

train_dataset = TensorDataset(train_inputs, train_targets)
eval_dataset = TensorDataset(eval_inputs, eval_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
