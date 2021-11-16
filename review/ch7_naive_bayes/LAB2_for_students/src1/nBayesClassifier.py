import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):
        '''
        需要你实现的部分
        '''
        # 得到Pc，记得取log
        labels = np.unique(trainlabel)
        D = traindata.shape[0]
        N = labels.shape[0]
        for label in labels:
            self.Pc[label] = np.log2((traindata[trainlabel.ravel() == label].shape[0] + 1) / (D + N))

        # 将训练集x用label分类成label个数个矩阵并存入字典
        data_groupby_c = {}
        for label in labels:
            data_groupby_c[label] = traindata[trainlabel.ravel() == label]

        for label in labels:
            self.Pxc[label] = {}
            for i in range(traindata.shape[1]):
                self.Pxc[label][i] = {}
                c_xi = data_groupby_c[label][:, i] # 得到分类后的数据特征的第i个特征列
                # 离散型特征，直接在Pxc中存入条件概率
                if featuretype[i] == 0:
                    c_xi_uniques = np.unique(c_xi)
                    Ni = c_xi_uniques.shape[0]
                    for c_xi_unique in c_xi_uniques:
                        self.Pxc[label][i][c_xi_unique] = np.log2(
                            (c_xi[c_xi == c_xi_unique].shape[0] + 1) / (c_xi.shape[0] + Ni))
                # 连续型特征，在Pxc中存入均值与方差
                else:
                    self.Pxc[label][i]['mu'] = np.mean(c_xi)
                    self.Pxc[label][i]['sigma2'] = np.var(c_xi, ddof=1)

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        '''
        需要你实现的部分
        '''

        # 定义高斯分布的概率密度函数
        def Gauss(x, mean, var):
            p_xc = 1 / (np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))
            return p_xc

        result = []
        for feature in features:
            logp = {} # logp中每个key表示本测试样例该key类的概率的log值
            for c in self.Pc.keys():
                logp[c] = self.Pc[c]
            for c in self.Pc.keys():
                for i in range(len(feature)):
                    # 离散型
                    if featuretype[i] == 0:
                        logp[c] = logp[c] + self.Pxc[c][i][feature[i]]
                    # 连续型
                    else:
                        mu = self.Pxc[c][i]['mu']
                        sigma2 = self.Pxc[c][i]['sigma2']
                        logp[c] = logp[c] + np.log2(Gauss(feature[i], mu, sigma2))
            result.append(max(logp, key=logp.get))
        # 结果转换为目标格式
        result = np.array([result]).T
        return result


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))
    # print(train_data[:10, :])
    # print(test_label[:20])


main()
