import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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
    data = pd.read_csv("..\\data\\Energy\\energydata_complete.csv")
    # data = pd.read_csv("C:\\Users\\HE CONG BING\\Desktop\\energydata_complete.csv")

    targets = data['Appliances'].values

    # print(targets.shape)
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(targets[0:5000], label='真实PM2.5含量', color='red', linewidth=2)
    # plt.xlabel('时间')
    # plt.ylabel('PM2.5含量')
    # plt.legend()
    # plt.show()

    t = np.arange(len(targets))
    targets_s = pd.DataFrame({'Time': t, 'Appliances': targets})

    lag_order = 2
    for i in range(1, lag_order + 1):
        targets_s[f'Value_Lag_{i}'] = targets_s['Appliances'].shift(i)

    targets_s['Value_Diff'] = targets_s['Appliances'].diff()
    targets_s = linear_interpolate(targets_s, 'Value_Lag_1')
    targets_s = linear_interpolate(targets_s, 'Value_Lag_2')
    targets_s = linear_interpolate(targets_s, 'Value_Diff')

    all_imformation = [
        data['Appliances'].tolist(),
        data['lights'].tolist(),
        data['T1'].tolist(),
        # data['RH_1'].tolist(),
        data['T2'].tolist(),
        data['RH_2'].tolist(),
        data['T3'].tolist(),
        # data['RH_3'].tolist(),
        data['T4'].tolist(),
        # data['RH_4'].tolist(),
        data['T5'].tolist(),
        # data['RH_5'].tolist(),
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
        # data['Visibility'].tolist(),
        # data['Tdewpoint'].tolist(),
        # data['rv1'].tolist(),
        # data['rv2'].tolist(),
        targets_s['Value_Lag_1'].tolist(),
        targets_s['Value_Lag_2'].tolist(),
        targets_s['Value_Diff'].tolist()
    ]

    seq_length = seq_len
    seq_half = seq_length // 2

    train_split = round(len(targets) * 0.80)
    test_split = round(len(targets) * 0.10)

    train_size = train_split
    eval_size = test_split
    test_size = test_split
    train_size_target = train_split + seq_length
    eval_size_target = train_split + test_split
    test_size_target = train_split + test_split + seq_length
    target_size = train_split + test_split + test_split

    num_data = [lst[:train_size + eval_size] for lst in all_imformation]
    all_normalize, target_normalize, targets_min, targets_max = normalize_data(num_data, all_imformation)
    # print(targets_min, targets_max)
    # air_normalize, PM25_normalize = all_imformation, PM25

    train_data = [lst[:train_size] for lst in all_normalize]
    eval_data = [lst[train_size:train_size + eval_size] for lst in all_normalize]
    test_data = [lst[train_size + eval_size:train_size + eval_size + test_size] for lst in all_normalize]

    train_sequences = create_sequences(train_data, seq_length, targets_len)
    eval_sequences = create_sequences(eval_data, seq_length, targets_len)
    test_sequences = create_sequences(test_data, seq_length, targets_len)
    train_sequences_s = train_sequences[:, 0, :]
    eval_sequences_s = eval_sequences[:, 0, :]
    test_sequences_s = test_sequences[:, 0, :]

    train_targets = target_normalize[seq_length:train_size]
    eval_targets = target_normalize[train_size_target:eval_size_target]
    test_targets = target_normalize[test_size_target:target_size]

    train_targets = create_target_sequences(train_targets, targets_len)
    eval_targets = create_target_sequences(eval_targets, targets_len)
    test_targets = create_target_sequences(test_targets, targets_len)

    train_inputs = torch.Tensor(train_sequences)
    eval_inputs = torch.Tensor(eval_sequences)
    test_inputs = torch.Tensor(test_sequences)
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
    # print(train_inputs.shape, eval_inputs.shape, test_inputs.shape, train_targets.shape, eval_targets.shape,
    #       test_targets.shape)

    train_dataset = TensorDataset(train_inputs, train_targets)
    eval_dataset = TensorDataset(eval_inputs, eval_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, eval_loader, test_loader, targets_min, targets_max


if __name__ == '__main__':
    data_loader(10, 1)
