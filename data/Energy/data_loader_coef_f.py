import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

data = pd.read_csv(".\\Energydata_2.csv")

PM25 = data['Appliances'].values

t = np.arange(len(PM25))
PM25_s = pd.DataFrame({'Time': t, 'Appliances': PM25})


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
    PM25_s[f'Value_Lag_{i}'] = PM25_s['Appliances'].shift(i)

PM25_s['Value_Diff'] = PM25_s['Appliances'].diff()
PM25_s = linear_interpolate(PM25_s, 'Value_Lag_1')
PM25_s = linear_interpolate(PM25_s, 'Value_Lag_2')
PM25_s = linear_interpolate(PM25_s, 'Value_Diff')

air_imformation = [
    data['Appliances'].tolist(),
    data['lights'].tolist(),
    data['T1'].tolist(),
    data['RH_1'].tolist(),
    data['T2'].tolist(),
    data['RH_2'].tolist(),
    data['T3'].tolist(),
    data['RH_3'].tolist(),
    data['T4'].tolist(),
    data['RH_4'].tolist(),
    data['T5'].tolist(),
    data['RH_5'].tolist(),
    data['T6'].tolist(),
    data['RH_6'].tolist(),
    data['T7'].tolist(),
    data['RH_7'].tolist(),
    data['T8'].tolist(),
    data['RH_8'].tolist(),
    data['T9'].tolist(),
    data['RH_9'].tolist(),
    data['T_out'].tolist(),
    data['Press_mm_hg'].tolist(),
    data['RH_out'].tolist(),
    data['Windspeed'].tolist(),
    data['Visibility'].tolist(),
    data['Tdewpoint'].tolist(),
    data['rv1'].tolist(),
    data['rv2'].tolist(),
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
        coef_expand = np.full_like(data_np[i, :], coef)
        data_output_np.append(coef_expand)
    output_data = torch.Tensor(np.array(data_output_np)).unsqueeze(dim=1)
    return output_data


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

train_targets = PM25_normalize[:train_size + eval_size]

train_data = np.array(train_data)
train_data = np.transpose(train_data)
train_targets = np.array(train_targets)
train_data_out = np.column_stack((train_data, train_targets))
print(train_data.shape, train_targets.shape, train_data_out.shape)