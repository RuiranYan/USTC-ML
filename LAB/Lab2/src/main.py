import numpy as np
import matplotlib.pyplot as plt
import Xgboost as xgb
from data_process import *
from evaluate import *

if __name__ == '__main__':
    path = '../data/train.data'
    path2 = '../data/ailerons.test'
    dataX, datay = load_data(path)
    print(f"shape of X is {dataX.shape}, shape of y is {datay.shape}")
    X_train, X_test, y_train, y_test = split_data(dataX, datay, 0.1)
    print(f"X_train:{X_train.shape}, X_test:{X_test.shape}, y_train:{y_train.shape}, y_test:{y_test.shape}")
    myXGB = xgb.XGboost(gamma=0.1,lambda_p=0.1)
    print("model fit begin")
    myXGB.fit(X_train,y_train)
    print("model fit finish")
    testdataX, testdatay = load_data(path2)
    y_pre = myXGB.predict(testdataX)
    y_test = testdatay
    print(abs(y_pre-y_test))
    print(f"RMSE is {get_RMSE(y_test,y_pre)}")
    print(f"R square is {get_Rsquare(y_test, y_pre)}")