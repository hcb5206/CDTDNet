import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

"""
20个值预测5个值
"""

data = pd.read_csv("C:\\Users\\HE CONG BING\\Desktop\\1001.csv")

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
    data['wind_direction'].tolist()
    # data['time_num'].tolist(),
    # data['PM25_average'].tolist(),
    # data['PM25_median'].tolist(),
    # data['PM25_max'].tolist(),
    # data['PM25_min'].tolist(),
    # data['PM25_seasonal'].tolist()
]


# data = pd.read_csv("C:\\Users\\HE CONG BING\\Desktop\\stock.csv")
#
# PM25 = data['收盘'].values
# air_imformation = [
#     data['收盘'].tolist(),
#     data['开盘'].tolist(),
#     data['高'].tolist(),
#     data['低'].tolist(),
#     data['涨跌幅'].tolist(),
#     data['交易量'].tolist(),
#     # data['weather'].tolist(),
#     # data['temperature'].tolist(),
#     # data['pressure'].tolist(),
#     # data['humidity'].tolist(),
#     # data['wind_speed'].tolist(),
#     # data['wind_direction'].tolist()
#     # data['time_num'].tolist(),
#     # data['PM25_average'].tolist(),
#     # data['PM25_median'].tolist(),
#     # data['PM25_max'].tolist(),
#     # data['PM25_min'].tolist()
# ]


def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    return data


def create_sequences(data, seq_length, target_length):
    sequences = []
    for i in range(len(data[0]) - seq_length - target_length + 1):
        seq = [lst[i:i + seq_length] for lst in data]
        sequences.append(seq)
        # if i == 0 or i == 1 or i == 6982 or i == 6983 or i == 851:
        #     print(i, seq)
    return np.array(sequences)


def create_target_sequences(data, target_length):
    sequences = []
    for i in range(len(data) - target_length + 1):
        seq = data[i:i + target_length]
        sequences.append(seq)
        # if i == 0 or i == 1 or i == 6982 or i == 6983 or i == 851:
        #     print(i, seq)
    return np.array(sequences)


air_normalize = normalize_data(air_imformation)
PM25_normalize = normalize_data(PM25)

seq_length = 10
target_length = 1

train_split = round(len(PM25) * 0.80)
test_split = round(len(PM25) * 0.10)
print(train_split, test_split)
train_size = train_split
eval_size = test_split
test_size = test_split
PM25_train_size = train_split + seq_length
PM25_eval_size = train_split + test_split
PM25_test_size = train_split + test_split + seq_length
PM25_size = train_split + test_split + test_split

train_data = [lst[:train_size] for lst in air_normalize]
eval_data = [lst[train_size:train_size + eval_size] for lst in air_normalize]
test_data = [lst[train_size + eval_size:train_size + eval_size + test_size] for lst in air_normalize]

train_sequences = create_sequences(train_data, seq_length, target_length)
eval_sequences = create_sequences(eval_data, seq_length, target_length)
test_sequences = create_sequences(test_data, seq_length, target_length)

train_targets_data = PM25_normalize[seq_length:train_size]
eval_targets_data = PM25_normalize[PM25_train_size:PM25_eval_size]
test_targets_data = PM25_normalize[PM25_test_size:PM25_size]

train_targets = create_target_sequences(train_targets_data, target_length)
eval_targets = create_target_sequences(eval_targets_data, target_length)
test_targets = create_target_sequences(test_targets_data, target_length)

train_inputs = torch.Tensor(train_sequences)
eval_inputs = torch.Tensor(eval_sequences)
test_inputs = torch.Tensor(test_sequences)
train_targets = torch.Tensor(train_targets)
eval_targets = torch.Tensor(eval_targets)
test_targets = torch.Tensor(test_targets)
print(train_inputs.shape, eval_inputs.shape, test_inputs.shape, train_targets.shape, eval_targets.shape, test_targets.shape)

train_dataset = TensorDataset(train_inputs, train_targets)
eval_dataset = TensorDataset(eval_inputs, eval_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
