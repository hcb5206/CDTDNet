import torch
import torch.nn as nn


def mse_loss(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2)


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))


def mse_loss_l2(y_pred, y_true):
    l2 = torch.norm((y_true - y_pred), p=2)
    return torch.mean(l2 ** 2)


def mae_loss_l1(y_pred, y_true):
    l1 = torch.norm((y_true - y_pred), p=1)
    return torch.mean(torch.abs(l1))


class CILLoss(nn.Module):
    def __init__(self, fs, smooth):
        super(CILLoss, self).__init__()
        self.fs = fs
        self.smooth = smooth
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def diff_2(self, x):
        diff = torch.diff(torch.diff(x))
        return diff

    def forward(self, predictions, targets):
        diff_2 = torch.sum((self.diff_2(predictions) - self.diff_2(targets)) ** 2)
        loss = self.fs * self.mse(predictions, targets) + (1 - self.fs) * self.mae(predictions,
                                                                                   targets) + self.smooth * diff_2
        return loss
