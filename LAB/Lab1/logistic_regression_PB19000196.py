import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys

# get data features
def load_data_features(datapath):
    features = []
    with open(datapath,'r') as f:
        for line in f.readlines():
            arr = []
            arr_str = line.strip().split(',')
            for i in range(2,len(arr_str)):
                arr.append(float(arr_str[i]))
            features.append(arr)
    return features

# get data labels
def load_data_labels(datapath):
    labels = []
    with open(datapath,'r') as f:
        for line in f.readlines():
            arr_str = line.strip().split(',')
            if arr_str[1] == 'M':
                label = 0
            else:
                label = 1
            labels.append(label)
    return labels

def sigmoid(x):
    return 1/(1+np.exp(-x))

# sigmoid一阶导数
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def loss(y_r,y_p): 
    eps = 1e-5
    return y_r*np.log(y_p+eps)+(1-y_r)*np.log(1-y_p+eps)

def loss_f(y_r, y_p): 
    return -np.sum(np.nan_to_num(y_r*np.log(y_p)+(1-y_r)*np.log(1-y_p)))/y_r.shape[0]

def sign(y):
    for i in range(y.shape[0]):
        if y[i]<0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y

def get_acc(y_r, y_p):
    y_p = sign(y_p)
    s = 0
    for i in range(y_r.shape[0]):
        if y_r[i]==y_p[i]:
            s+=1
    return s/y_r.shape[0]

def int2str(y_int):
    y_str = []
    for y in y_int:
        if y==0:
            y_str.append('M')
        else:
            y_str.append('B')
    return y_str

def train(X, y,itr,lr): # X:(n,31),y:(n,)
    X=np.array(X)
    n,m = X.shape
    y = np.array(y)
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X=(X-mu)/sigma
    X = np.c_[X,np.ones((n,1))]
    w = np.random.randn(m+1)
    for i in range(itr):
        y_p = sigmoid(np.dot(w,X.transpose()))
        dw = np.dot((y_p-y),X)/n
        w -= lr*dw
    y_p = sigmoid(np.dot(w,X.transpose()))
    return w, mu,sigma

def get_result(X, w, mu, sigma):
    X=np.array(X)
    n,m = X.shape
    X=(X-mu)/sigma
    X = np.c_[X,np.ones((n,1))]
    y_p = sigmoid(np.dot(w,X.transpose()))
    y_p = sign(y_p)
    return int2str(y_p)


if __name__ =='__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    X_train = load_data_features(train_path)
    y_train = load_data_labels(train_path)
    X_test = load_data_features(test_path)
    w, mu, sigma = train(X_train, y_train, 5000, 0.01)
    results = get_result(X_test, w, mu, sigma)
    for result in results:
        print(result)
