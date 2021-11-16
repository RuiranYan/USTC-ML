from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.05, Lambda=0.001, epochs=1000):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs
        self.w = None

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''

    def fit(self, train_features, train_labels):
        ''''
        需要你实现的部分
        '''

        X = np.c_[np.ones(train_features.shape[0]), train_features]
        y = train_labels
        labels = np.unique(y)

        def convert_to_one_hot(y, C):
            return np.eye(C)[y.reshape(-1)]

        labels_index = np.arange(len(labels))
        y_onehot = convert_to_one_hot(y-1, len(labels_index))  # onehot编码
        self.w = np.random.random((X.shape[1], len(labels_index)))
        m = train_features.shape[0]
        for _ in range(self.epochs):
            dw = 1 / m * np.dot(X.T, (np.dot(X, self.w) - y_onehot)) + self.Lambda * self.w
            self.w = self.w - self.lr * dw

            '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
            预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''

    def predict(self, test_features):
        ''''
        需要你实现的部分
        '''
        x = np.c_[np.ones(test_features.shape[0]), test_features]
        y = np.dot(x, self.w)
        result = np.argmax(y, axis=1)+1
        result = np.array([result]).T
        return result




def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
