import numpy as np
import cvxopt  # 用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


# 根据指定类别main_class生成1/-1标签
def svm_label(labels, main_class):
    new_label = []
    for i in range(len(labels)):
        if labels[i] == main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)


# 实现线性回归
class SupportVectorMachine:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, kernel, C, Epsilon):
        self.kernel = kernel
        self.C = C
        self.Epsilon = Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''

    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        # d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2)**2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1, x2)
        elif kernel == 'Poly':
            K = np.dot(x1, x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''

    def fit(self, train_data, train_label, test_data):
        '''
        需要你实现的部分
        '''
        # fit
        m = train_data.shape[0]
        k = np.zeros((train_data.shape[0], train_data.shape[0]))
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[0]):
                k[i, j] = self.KERNEL(train_data[i], train_data[j], self.kernel)
        P = cvxopt.matrix(np.outer(train_label, train_label) * k)
        q = cvxopt.matrix(-1.0 * np.ones(train_data.shape[0]))
        A = cvxopt.matrix(train_label.astype(float), (1, len(train_label)))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.r_[-1.0 * np.eye(m), np.eye(m)])
        h = cvxopt.matrix(np.r_[np.zeros((m, 1)), self.C * np.ones((m, 1))])
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])
        print("alpha:", alpha)
        support_vector_tf = []
        for i in range(len(alpha)):
            support_vector_tf.append(self.Epsilon < alpha[i] < self.C)
        support_vector_alpha = alpha[support_vector_tf]
        support_vector_index = np.arange(len(alpha))[support_vector_tf]
        support_vector_x = train_data[support_vector_tf]
        support_vector_y = train_label[support_vector_tf]
        # print("sv", support_vector_alpha)
        b = 0
        # b = avg(b_i)
        for i in range(len(support_vector_alpha)):
            b = b + support_vector_y[i]
            b = b - np.sum(support_vector_alpha * support_vector_y * k[support_vector_index[i], support_vector_tf])
        b /= len(support_vector_alpha)

        # predict
        pred = []
        for x in test_data:
            s = 0
            for train_data_i, train_label_i, alpha_i in zip(train_data, train_label, alpha):
                s = s + alpha_i * train_label_i * self.KERNEL(x, train_data_i, self.kernel)
            pred.append(s + b)
        # print(pred)
        return pred



def main():
    # 加载训练集和测试集
    Train_data, Train_label, Test_data, Test_label = load_and_process_data()
    Train_label = [label[0] for label in Train_label]
    Test_label = [label[0] for label in Test_label]
    train_data = np.array(Train_data)
    test_data = np.array(Test_data)
    test_label = np.array(Test_label).reshape(-1, 1)
    # 类别个数
    num_class = len(set(Train_label))

    # kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    # C为软间隔参数；
    # Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel = 'Linear'
    C = 1
    Epsilon = 10e-5
    # 生成SVM分类器
    SVM = SupportVectorMachine(kernel, C, Epsilon)

    predictions = []
    # one-vs-all方法训练num_class个二分类器
    for k in range(1, num_class + 1):
        # 将第k类样本label置为1，其余类别置为-1
        train_label = svm_label(Train_label, k)
        # 训练模型，并得到测试集上的预测结果
        prediction = SVM.fit(train_data, train_label, test_data)
        # print('sh', len(prediction))
        predictions.append(prediction)
    predictions = np.array(predictions)
    # print(predictions)
    print('shape:', predictions.shape)
    # one-vs-all, 最终分类结果选择最大score对应的类别
    pred = np.argmax(predictions, axis=0) + 1
    pred = pred.reshape(-1, 1)

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
