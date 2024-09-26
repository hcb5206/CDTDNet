import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

data = pd.read_csv("C:\\Users\\HE CONG BING\\Desktop\\1001.csv")

PM25 = data['PM25_Concentration'].values
# air_imformation = [
#     data['PM25_Concentration'].tolist()
    # data['PM10_Concentration'].tolist(),
    # data['NO2_Concentration'].tolist(),
    # data['CO_Concentration'].tolist(),
    # data['O3_Concentration'].tolist(),
    # data['SO2_Concentration'].tolist(),
    # data['weather'].tolist(),
    # data['temperature'].tolist(),
    # data['pressure'].tolist(),
    # data['humidity'].tolist(),
    # data['wind_speed'].tolist(),
    # data['wind_direction'].tolist(),
    # data['time_num'].tolist()
    # data['PM25_average'].tolist(),
    # data['PM25_median'].tolist(),
    # data['PM25_max'].tolist(),
    # data['PM25_min'].tolist(),
    # data['PM25_var'].tolist(),
    # data['PM25_std'].tolist(),
    # data['PM25_seasonal'].tolist()
# ]


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


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data[0]) - seq_length):
        seq = [lst[i:i + seq_length] for lst in data]
        sequences.append(seq)
    return np.array(sequences)


def create_sequences_s(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


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

    concatenated_tensor = torch.cat(
        (data_tensor, mean_filled_data_tensor, median_filled_data_tensor, max_filled_data_tensor,
         min_filled_data_tensor, std_filled_data_tensor), dim=1)
    # print(concatenated_tensor.size())
    # print(data_tensor[1, :, :])
    # print(mean_value_first_half[1, :, :])
    # print(mean_value_second_half[1, :, :])
    # print(concatenated_tensor[1, :, :])
    return concatenated_tensor


# air_normalize = normalize_data(air_imformation)
PM25_normalize = PM25
# air_normalize = air_imformation
# PM25_normalize = PM25
seq_length = 10

train_split = round(len(PM25) * 0.80)
test_split = round(len(PM25) * 0.10)
# print(train_split, test_split)

train_size = train_split
eval_size = test_split
test_size = test_split
PM25_train_size = train_split + seq_length
PM25_eval_size = train_split + test_split
PM25_test_size = train_split + test_split + seq_length
PM25_size = train_split + test_split + test_split

# train_data = [lst[:train_size] for lst in air_normalize]
# eval_data = [lst[train_size:train_size + eval_size] for lst in air_normalize]
# test_data = [lst[train_size + eval_size:train_size + eval_size + test_size] for lst in air_normalize]

train_data_s = PM25_normalize[:train_size]
eval_data_s = PM25_normalize[train_size:train_size + eval_size]
test_data_s = PM25_normalize[train_size + eval_size:train_size + eval_size + test_size]
# print(train_data, '*' * 100)
# print(eval_data, '*' * 100)
# print(test_data, '*' * 100)

# train_sequences = create_sequences(train_data, seq_length)
# eval_sequences = create_sequences(eval_data, seq_length)
# test_sequences = create_sequences(test_data, seq_length)
train_sequences_s = create_sequences_s(train_data_s, seq_length)
eval_sequences_s = create_sequences_s(eval_data_s, seq_length)
test_sequences_s = create_sequences_s(test_data_s, seq_length)

train_sequences_s = normalize_data(train_sequences_s)
eval_sequences_s = normalize_data(eval_sequences_s)
test_sequences_s = normalize_data(test_sequences_s)

train_targets = PM25_normalize[seq_length:train_size]
eval_targets = PM25_normalize[PM25_train_size:PM25_eval_size]
test_targets = PM25_normalize[PM25_test_size:PM25_size]
# print(train_sequences)
# print(train_targets)

# train_inputs = torch.Tensor(train_sequences)
# eval_inputs = torch.Tensor(eval_sequences)
# test_inputs = torch.Tensor(test_sequences)
train_inputs_s = torch.Tensor(train_sequences_s).unsqueeze(dim=1)
eval_inputs_s = torch.Tensor(eval_sequences_s).unsqueeze(dim=1)
test_inputs_s = torch.Tensor(test_sequences_s).unsqueeze(dim=1)
train_targets = torch.Tensor(train_targets)
eval_targets = torch.Tensor(eval_targets)
test_targets = torch.Tensor(test_targets)

# print(train_inputs_s.shape, eval_inputs_s.shape, test_inputs_s.shape)
train_inputs = time_imformation(train_inputs_s)
eval_inputs = time_imformation(eval_inputs_s)
test_inputs = time_imformation(test_inputs_s)

# print(train_inputs[10, :, :])
# print(train_inputs_s[10, :, :])
# print(train_inputs.shape, train_inputs_s.shape, eval_inputs.shape, eval_inputs_s.shape, test_inputs.shape,
#       test_inputs_s.shape)
# train_inputs = torch.cat((train_inputs, train_inputs_s), dim=1)
# eval_inputs = torch.cat((eval_inputs, eval_inputs_s), dim=1)
# test_inputs = torch.cat((test_inputs, test_inputs_s), dim=1)
# print(train_inputs[10, :, :])
# print(train_inputs[:, 0, :])
# print(train_targets)
print(train_inputs.shape, eval_inputs.shape, test_inputs.shape, train_targets.shape, eval_targets.shape,
      test_targets.shape)

train_dataset = TensorDataset(train_inputs, train_targets)
eval_dataset = TensorDataset(eval_inputs, eval_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
