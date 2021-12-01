import numpy as np
import matplotlib.pyplot as plt
import Xgboost as xgb
from data_process import *
from evaluate import *
from sklearn import linear_model


def test():
    # init param
    gamma, lambda_p, max_depth, m = 1e-6, 1, 3, 10
    # data path
    path = '../data/train.data'
    path2 = '../data/ailerons.test'
    # get data
    dataX, datay = load_data(path)
    # show data shape
    print(f"shape of X is {dataX.shape}, shape of y is {datay.shape}")
    X_train, X_test, y_train, y_test = split_data(dataX, datay, 0.1)
    print(f"X_train:{X_train.shape}, X_test:{X_test.shape}, y_train:{y_train.shape}, y_test:{y_test.shape}")
    # init model
    myXGB = xgb.XGboost(gamma=gamma, lambda_p=lambda_p, max_depth=max_depth, m=m)
    print("model fit begin")
    myXGB.fit(X_train, y_train)
    print("model fit finish")
    # # verify
    # y_pre = myXGB.predict(X_test)
    # myXGB.draw_pic(X_test, y_test)
    testdataX, testdatay = load_data(path2)  # get test set
    myXGB.draw_pic(testdataX, testdatay)
    y_pre = myXGB.predict(testdataX)
    # compare with linear model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pre_linear = model.predict(testdataX)
    print("xgboost:")
    print(f"RMSE is {get_RMSE(testdatay, y_pre)}")
    print(f"R square is {get_Rsquare(testdatay, y_pre)}")
    print("linear regression:")
    print(f"RMSE is {get_RMSE(testdatay, y_pre_linear)}")
    print(f"R square is {get_Rsquare(testdatay, y_pre_linear)}")


if __name__ == '__main__':
    test()
