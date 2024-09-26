import pandas as pd
import numpy as np
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


def normalize_data(data, data_all):
    data_train = []
    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        data_s = (data_all[i] - min_val) / (max_val - min_val)
        data_train.append(data_s)
    return data_train


def normalize_data_targets(data, data_all):
    min_val = np.min(data)
    max_val = np.max(data)
    data_s = (data_all - min_val) / (max_val - min_val)
    return data_s


def create_sequences_targets(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
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

num_data = [lst[:train_size + eval_size] for lst in air_imformation]
air_normalize = normalize_data(num_data, air_imformation)
PM_num_data = PM25[:train_size + eval_size]
PM25_normalize = normalize_data_targets(PM_num_data, PM25)
# air_normalize, PM25_normalize = air_imformation, PM25

train_data = [lst[:train_size - seq_length] for lst in air_normalize]
test_data = [lst[train_size + eval_size:train_size + eval_size + test_size - seq_length] for lst in air_normalize]
train_targets = PM25_normalize[seq_length:train_size]
test_targets = PM25_normalize[train_size + eval_size + seq_length:train_size + eval_size + test_size]

train_targets_s = PM25_normalize[:train_size]
test_targets_s = PM25_normalize[train_size + eval_size:train_size + eval_size + test_size]

train_sequences = create_sequences_targets(train_targets_s, seq_length)
test_sequences = create_sequences_targets(test_targets_s, seq_length)

train_inputs_s = torch.Tensor(train_sequences).unsqueeze(dim=1)
test_inputs_s = torch.Tensor(test_sequences).unsqueeze(dim=1)

train_inputs_s = time_imformation(train_inputs_s)
test_inputs_s = time_imformation(test_inputs_s)

train_inputs_s = np.array(train_inputs_s)
test_inputs_s = np.array(test_inputs_s)
train_data = np.array(train_data)
test_data = np.array(test_data)
test_data = np.transpose(test_data)
train_data = np.transpose(train_data)
train_data = np.column_stack((train_data, train_inputs_s))
test_data = np.column_stack((test_data, test_inputs_s))
train_targets = np.array(train_targets)
test_targets = np.array(test_targets)
print(train_data.shape, train_targets.shape, test_data.shape, test_targets.shape)
