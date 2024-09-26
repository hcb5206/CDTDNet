import torch
import torch.nn.functional as F


def root_mean_squared_error(y_true, y_pred):
    mse = F.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return rmse.item()


def mean_squared_error(y_true, y_pred):
    mse = F.mse_loss(y_pred, y_true)
    return mse.item()


def mean_absolute_error(y_true, y_pred):
    mae = F.l1_loss(y_pred, y_true)
    return mae.item()


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    absolute_percentage_errors = torch.abs((y_true - y_pred) / (y_true + y_pred))
    smape = 2 * torch.mean(absolute_percentage_errors) * 100
    return smape.item()


def coefficient_of_determination(y_true, y_pred):
    mean_y_true = torch.mean(y_true)
    total_sum_of_squares = torch.sum((y_true - mean_y_true) ** 2)
    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2.item()


def NormalizedAbsoluteError(y_true, y_pred):
    nae = torch.mean(torch.abs(y_pred - y_true) / (torch.max(torch.abs(y_true), torch.abs(y_pred)) + 1e-8))
    return nae.item()


def index_of_agreement(y_true, y_pred):
    y_mean = torch.mean(y_true)
    num = torch.sum((y_true - y_pred) ** 2)
    denom = torch.sum((torch.abs(y_pred - y_mean) + torch.abs(y_true - y_mean)) ** 2)

    ia = 1 - (num / denom)
    return ia.item()
