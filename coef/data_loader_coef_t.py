import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

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
    data['wind_direction'].tolist(),
    data['time_num'].tolist(),
    data['date_num'].tolist()
]


# def normalize_data(data):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     data = (data - min_val) / (max_val - min_val)
#     return data

def normalize_data(data, data_all):
    data_train = []
    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        data_s = (data_all[i] - min_val) / (max_val - min_val)
        data_train.append(data_s)
        if i == 0:
            data_target = data_s
    return data_train, data_target


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
        # coef_expand = np.full_like(data_np[i, :], coef)
        data_output_np.append(coef)
    output_data = torch.Tensor(np.array(data_output_np)).unsqueeze(dim=1)
    return output_data


# def create_sequences_s(data, seq_length):
#     sequences = []
#     for i in range(len(data) - seq_length):
#         seq = data[i:i + seq_length]
#         sequences.append(seq)
#     return np.array(sequences)


def time_imformation(data_tensor):
    mean_filled_data_tensor = data_tensor.mean(dim=2, keepdim=True).squeeze(dim=2)

    median_filled_data_tensor = data_tensor.median(dim=2, keepdim=True)[0].squeeze(dim=2)

    max_filled_data_tensor = data_tensor.max(dim=2, keepdim=True)[0].squeeze(dim=2)

    min_filled_data_tensor = data_tensor.min(dim=2, keepdim=True)[0].squeeze(dim=2)

    std_filled_data_tensor = data_tensor.std(dim=2, keepdim=True).squeeze(dim=2)
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
air_normalize, PM25_normalize = normalize_data(num_data, air_imformation)
# air_normalize, PM25_normalize = air_imformation, PM25

train_data = [lst[:train_size + eval_size] for lst in air_normalize]

train_sequences = create_sequences(train_data, seq_length)

train_sequences_s = train_sequences[:, 0, :]

train_targets = PM25_normalize[seq_length:train_size + eval_size]

train_inputs_s = torch.Tensor(train_sequences_s).unsqueeze(dim=1)

train_targets = torch.Tensor(train_targets)

train_inputs_s = time_imformation(train_inputs_s)

train_inputs_s = np.array(train_inputs_s)
train_targets_s = np.array(train_targets)
train_data_out_s = np.column_stack((train_inputs_s, train_targets))
print(train_inputs_s.shape, train_targets_s.shape, train_data_out_s.shape)