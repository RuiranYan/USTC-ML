import numpy as np
import matplotlib.pyplot as plt
from evaluate import get_RMSE, get_Rsquare


class XGboost(object):
    def __init__(self,
                 gamma=0,
                 lambda_p=0,
                 max_depth=3,  # 最大深度
                 m=10  # 子树个数
                 ):
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.max_depth = max_depth
        self.m = m
        self.TreeList = []

    def fit(self, X, y):
        if len(self.TreeList) != 0:
            self.TreeList = []
        y_t = np.zeros(y.shape)
        data = np.c_[X, y, y_t]
        for i in range(self.m):
            print(f"No.{i + 1} tree is building ......")
            tree = DecisionTree(data, self.gamma, self.lambda_p, self.max_depth)
            self.TreeList.append(tree)
            print(f"No.{i + 1} tree has built,wait for the next tree ......")
            data[:, -1:] = self.predict(X)

    def predict(self, X):
        # X:(N,40)
        if len(self.TreeList) == 0:
            print("TreeList is empty, you need to fit data first")
        else:
            n, _ = X.shape
            y_pre = np.zeros((n, 1))
            for i in range(n):
                for tree in self.TreeList:
                    y_pre[i, 0] += tree.inference(X[i])
            return y_pre

    def get_loss(self, X, y):
        if len(self.TreeList) == 0:
            print("TreeList is empty, you need to fit data first")
        else:
            n, _ = X.shape
            y_pre = np.zeros((n, 1))
            losslist = []
            for i in range(len(self.TreeList)):
                for tree in self.TreeList[:i]:
                    for j in range(n):
                        y_pre[j, 0] += tree.inference(X[j])
                loss = np.sum((y - y_pre) ** 2)
                y_pre = np.zeros((n, 1))
                losslist.append(loss)
            return np.array(losslist)

    def draw_pic(self, X, y):
        losslist = self.get_loss(X, y)
        y_pre = self.predict(X)
        rmse = get_RMSE(y, y_pre)
        r2 = get_Rsquare(y, y_pre)
        plt.plot(losslist)
        plt.xlabel("tree number")
        plt.ylabel("loss")
        plt.title(
            f"gamma={self.gamma}, lambda={self.lambda_p}, max_depth={self.max_depth}, m={self.m}\nrmse = {rmse}, r2 = {r2}")
        plt.show()


class Node(object):
    def __init__(self):
        self.l = None
        self.r = None
        self.feature = None
        self.f_value = None
        self.isleaf = False
        self.depth = None
        self.omega = None


class DecisionTree(object):
    def __init__(self, data, gamma, lambda_p, max_depth=20):
        # data最后一列为y_t，shape = (n,42)
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.max_depth = max_depth
        self.feature = None
        self.f_value = None
        self.root = self.createTree(data, 0)

    def get_obj1(self, G, H):
        return 0.5 * G ** 2 / (H + self.lambda_p) + self.gamma

    def get_obj2(self, Gl, Hl, Gr, Hr):
        return 0.5 * (Gl ** 2 / (Hl + self.lambda_p) + Gr ** 2 / (Hr + self.lambda_p)) + 2 * self.gamma

    def get_omega(self, data):
        n, F = data.shape
        y_t, y = data[:, -1], data[:, -2]
        G = np.sum(-2 * (y - y_t))
        H = 2 * n
        return -G / (H + self.lambda_p)

    def createTree(self, data, depth):
        if depth < self.max_depth:
            root = Node()
            root.depth = depth
            # find split
            n, F = data.shape
            F -= 2
            y_t, y = data[:, -1:], data[:, -2:-1]
            G = np.sum(-2 * (y - y_t))
            H = 2 * n
            obj1 = self.get_obj1(G, H)
            max_gain = 0
            for feature in range(F):
                tmp = np.c_[data[:, feature:feature + 1], -2 * (y - y_t)]
                sorted_f_value_list = tmp[np.argsort(tmp[:, 0])]
                Gl, Gr, Hl, Hr = 0, G, 0, H
                for i in range(sorted_f_value_list.shape[0]):
                    # 小于等于i时划分到左侧
                    Gl += sorted_f_value_list[i, -1]
                    Gr = G - Gl
                    Hl += 2
                    Hr = H - Hl
                    obj2 = self.get_obj2(Gl, Hl, Gr, Hr)
                    gain = obj2 - obj1
                    if gain > max_gain:
                        max_gain = gain
                        self.feature, self.f_value = feature, sorted_f_value_list[i, 0]
            root.feature, root.f_value = self.feature, self.f_value
            data_l = data[data[:, self.feature] <= self.f_value, :]
            data_r = data[data[:, self.feature] > self.f_value, :]
            root.l = self.createTree(data_l, depth + 1)
            root.r = self.createTree(data_r, depth + 1)
            return root
        else:
            leaf = Node()
            leaf.depth = depth
            leaf.isleaf = True
            leaf.omega = self.get_omega(data)
            return leaf

    def inference(self, x):
        p = self.root
        while not p.isleaf:
            if x[p.feature] <= p.f_value:
                p = p.l
            elif x[p.feature] > p.f_value:
                p = p.r
        return p.omega
