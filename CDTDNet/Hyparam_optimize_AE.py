# from data_loader_s_fac import data_loader, denormalize
from data_loader.data_loader_AirQuality import data_loader, denormalize
# from data_loader.data_loader_Energy import data_loader, denormalize
# from data_loader.data_loader_traffic_wavelet import data_loader, denormalize
# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize
from evaluate import mean_absolute_error, symmetric_mean_absolute_percentage_error, \
    mean_squared_error, root_mean_squared_error, coefficient_of_determination, \
    NormalizedAbsoluteError, index_of_agreement
import CDTDNet
from Loss_Function import CILLoss
import torch
import torch.nn as nn
import torch.optim as optim
from skopt import gp_minimize, space
import math
import os

"""
AIR:
Best Parameters: [500, 64, 0.05, 0.1, 0.02, 40] | Best Value: 14.731707829695482

Energy:MSE
Best Parameters: [359, 16, 0.02, 160] | Best Value: 62.54897435506185

NFLX:
Best Parameters: [315, 128, 0.05, 0.1, 0.005, 120] | Best Value: 7.537879467010498

Traffic:
Best Parameters: [500, 128, 0.2, 0.05, 0.002, 40] | Best Value: 0.045954830944538116
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

model_path = 'models/Air/Air_SEQ_LSTM_1S'


def cul_layers(seq_len, b, k, hidden_size, out_size):
    out_l = []
    s = ((seq_len - 1) * (b - 1) / (k - 1) + 1)
    n = math.ceil(math.log(s, b))
    for i in range(n):
        if i < n - 1:
            out_l.append(hidden_size)
        else:
            out_l.append(out_size)

    return out_l


def objective_function(params):
    seed_s, hidden_lag, fs, smooth, learn_rate, num_epochs = params
    torch.manual_seed(seed_s)
    seq_len = 10
    targets_len = 1
    input_size = 23
    input_tcn = 17
    batch_size = 64
    hidden_size = 32
    flavor = 'BiCSCRU'
    kernel_size = 3
    hidden_tcn = 64
    dropout = 0.5
    hidden_att_f = 64

    input_dim = input_size + 1
    output_dim = 1

    train_loader, eval_loader, test_loader, targets_min, targets_max = data_loader(seq_len, targets_len)
    num_channels = cul_layers(seq_len, b=2, k=kernel_size, hidden_size=hidden_tcn, out_size=input_tcn)
    model = CDTDNet.VAE(input_dim=input_dim, hidden_dim=hidden_lag, output_dim=output_dim)
    # model.to(device)
    model_seq = CDTDNet.EncoderDecoder(input_size=input_size, hidden_size=hidden_size, input_tcn=input_tcn,
                                           npred=targets_len, flavor=flavor, seq_len=seq_len, num_channels=num_channels,
                                           kernel_size=kernel_size, dropout=dropout, hidden_att_f=hidden_att_f,
                                           batch_size=batch_size)
    model_seq.load_state_dict(torch.load(model_path))
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    criterion = CILLoss(fs=fs, smooth=smooth)
    optimizer = optim.Adagrad(model.parameters(), lr=learn_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    for _ in range(num_epochs):
        model.train()
        model_seq.eval()
        for inputs, targets in train_loader:
            targets = targets.squeeze()
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            outputs_seq = model_seq(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])

            outputs_seq = outputs_seq.unsqueeze(dim=1).unsqueeze(dim=2)

            # outputs_seq = outputs_seq.unsqueeze(dim=2)

            inputs_seq = inputs[:, :, -targets_len:].transpose(1, 2)
            outputs_seq = torch.cat((outputs_seq, inputs_seq), dim=2)

            # print(inputs_seq.shape, outputs_seq.shape)
            # outputs_seq = outputs_seq.to(device)
            # targets = targets.to(device)
            outputs = model(outputs_seq).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    model_seq.eval()
    eval_rmse = 0.0
    with torch.no_grad():
        for inputs_eval, targets_eval in eval_loader:
            targets_eval = targets_eval.squeeze()
            outputs_eval = model_seq(inputs_eval[:, input_tcn:, :], inputs_eval[:, :input_tcn, :])

            outputs_seq = outputs_eval.unsqueeze(dim=1).unsqueeze(dim=2)

            # outputs_seq = outputs_eval.unsqueeze(dim=2)

            inputs_seq = inputs_eval[:, :, -targets_len:].transpose(1, 2)
            outputs_seq = torch.cat((outputs_seq, inputs_seq), dim=2)

            # outputs_seq = outputs_seq.to(device)
            outputs_seq_eval = model(outputs_seq)
            outputs_eval = denormalize(outputs_seq_eval, targets_min, targets_max)
            targets_eval = denormalize(targets_eval, targets_min, targets_max)
            rmse = root_mean_squared_error(targets_eval.squeeze(), outputs_eval.squeeze())
            eval_rmse += rmse
        eval_rmse /= len(test_loader)
    print(
        f'eval RMSE:{eval_rmse:.4f}, seed:{seed_s}, hidden_lag:{hidden_lag}, '
        f'fs:{fs}, smooth:{smooth}, learn_rate:{learn_rate}, num_epochs:{num_epochs}')

    return eval_rmse


space = [
    space.Integer(0, 500, name='seed_s'),
    space.Categorical([16, 32, 64, 128], name='hidden_lag'),
    space.Categorical([0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5], name='fs'),
    space.Categorical([0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5], name='smooth'),
    space.Categorical([0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001], name='learn_rate'),
    space.Categorical([40, 50, 60, 70, 80, 90, 100, 120, 140, 160], name='num_epochs')
]


def print_result(res):
    iteration = len(res.func_vals)
    best_value = res.fun
    best_params = res.x
    print(f"Iteration {iteration}: Best Parameters: {best_params} | Best Value: {best_value}")


result = gp_minimize(objective_function, space, n_calls=50, callback=print_result)

best_params = result.x
print("Best Parameters:", best_params)

# if __name__ == '__main__':
#     objective_function()
