import numpy as np
import matplotlib.pyplot as plt


# 读取路径中的数据
def load_data(datapath):
    features = []
    labels = []
    with open(datapath, 'r') as f:
        for line in f.readlines():
            arrX = []
            arry = []
            arr_str = line.strip().split(',')
            for i in range(len(arr_str) - 1):
                arrX.append(float(arr_str[i]))
            arry.append(float(arr_str[-1]))
            features.append(arrX)
            labels.append(arry)
    return np.array(features), np.array(labels)


# 划分数据集，test_ratio为测试集所占比例
def split_data(dataX, datay, test_ratio, random_seed=101):
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(dataX))
    test_set_size = int(len(dataX) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataX[train_indices], dataX[test_indices], datay[train_indices], datay[test_indices]
