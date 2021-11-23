import numpy as np
from sklearn.metrics import r2_score


def get_acc(y_true, y_pre):
    return np.sum(y_true == y_pre) / y_true.size


def get_MSE(y_true, y_pre):
    return np.sum((y_true - y_pre) ** 2) / y_true.size


def get_RMSE(y_true, y_pre):
    return (get_MSE(y_true,y_pre)) ** 0.5



def get_Rsquare(y_true, y_pre):
    return r2_score(y_true,y_pre)