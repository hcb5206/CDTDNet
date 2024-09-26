from data_loader.data_loader_AirQuality import data_loader, denormalize
# from data_loader.data_loader_AirQuality_Gaussian import data_loader, denormalize
# from data_loader.data_loader_Energy import data_loader, denormalize
# from data_loader.data_loader_traffic_wavelet import data_loader, denormalize
# from data_loader.data_loader_NFLX_wavelet import data_loader, denormalize
from evaluate import root_mean_squared_error
from CDTDNet import EncoderDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from skopt import gp_minimize, space
import math
from Loss_Function import CILLoss
from functools import lru_cache
from skopt.space import Real, Integer
from hyperopt import hp, space_eval, tpe
from hyperopt.pyll import scope

"""
AirQuality:
Best Parameters: [381, 10, 32, 3, 64, 0.5, 64, 0.001, 150]
Energy:
Best Parameters: MSE:[232, 5, 64, 2, 128, 0.2, 32, 0.0002, 180]
NFLX:
Best Parameters: [500, 5, 128, 5, 32, 0.6, 128, 0.0005, 200]
traffic:
Best Parameters: [192, 29, 64, 4, 64, 0.1, 128, 0.0001, 200]
"""


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


# @lru_cache(maxsize=None)
def objective_function(params):
    seed_s, seq_len, hidden_de, kernel_size, hidden_tcn, dropout, hidden_att_f, learn_rate, num_epochs = params
    torch.manual_seed(seed_s)
    # seq_len = 12
    input_size = 23
    input_tcn = 17
    hidden_size = hidden_de
    flavor = 'BiCSCRU'
    kernel_size = kernel_size
    hidden_tcn = hidden_tcn
    dropout = dropout
    hidden_att_f = hidden_att_f
    targets_len = 1
    batch_size = 64
    # noise_std = 0
    train_loader, eval_loader, test_loader, targets_min, targets_max = data_loader(seq_len, targets_len)
    num_channels = cul_layers(seq_len=seq_len, b=2, k=kernel_size, hidden_size=hidden_tcn, out_size=input_tcn)
    model = EncoderDecoder(input_size=input_size, hidden_size=hidden_size, input_tcn=input_tcn, npred=targets_len,
                           flavor=flavor, seq_len=seq_len, num_channels=num_channels, kernel_size=kernel_size,
                           dropout=dropout, hidden_att_f=hidden_att_f, batch_size=batch_size)
    # model.to(device)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = CILLoss(fs=fs)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    eval_rmse = 0.0
    for _ in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            outputs = model(inputs[:, input_tcn:, :], inputs[:, :input_tcn, :])
            # outputs = model(inputs, inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        for inputs_eval, targets_eval in eval_loader:
            outputs_eval = model(inputs_eval[:, input_tcn:, :], inputs_eval[:, :input_tcn, :])
            outputs_eval = denormalize(outputs_eval.squeeze(), targets_min, targets_max)
            targets_eval = denormalize(targets_eval.squeeze(), targets_min, targets_max)
            rmse = root_mean_squared_error(targets_eval.squeeze(), outputs_eval.squeeze())
            eval_rmse += rmse
        eval_rmse /= len(test_loader)
    print(
        f'Test RMSE:{eval_rmse:.4f}, seed:{seed_s}, seq_len:{seq_len}, hidden_de:{hidden_size}, kernel_size:{kernel_size}, hidden_tcn:{hidden_tcn}, '
        f'dropout:{dropout}, hidden_att_f:{hidden_att_f}, learn_rate:{learn_rate}, num_epochs:{num_epochs}')

    return eval_rmse


space = [
    space.Integer(0, 500, name='seed_s'),
    space.Integer(5, 35, name='seq_len'),
    space.Categorical([16, 32, 64, 128], name='hidden_de'),
    space.Categorical([2, 3, 4, 5], name='kernel_size'),
    space.Categorical([16, 32, 64, 128], name='hidden_tcn'),
    space.Categorical([0.1, 0.2, 0.3, 0.4, 0.5], name='dropout'),
    space.Categorical([16, 32, 64, 128], name='hidden_att_f'),
    space.Categorical([0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001], name='learn_rate'),
    space.Categorical([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200], name='num_epochs')
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
