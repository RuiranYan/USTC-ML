import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score

color = {
    -2: 'black',
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'brown',
    4: 'pink',
    5: 'yellow',
    6: 'purple',
    7: 'orange',
    8: 'cyan',
    9: 'gold',
    10: 'gray'
}


class DPC(object):
    def __init__(self, dc, k='sign'):
        self.dc = dc
        self.k = k
        self.dij = None
        self.density = None
        self.delta = None
        self.thr_den = None
        self.thr_delt = None
        self.parent_index = None
        self.cluster_list = None

    def distance(self, x1, x2):
        return np.sum((x1 - x2) ** 2) ** 0.5

    def get_dij(self, X):
        n, m = X.shape
        self.dij = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.dij[i, j] = self.distance(X[i], X[j])
        # print("dij:")
        # print(self.dij)
        return self.dij

    def get_density_i(self, X):
        if self.dij is None:
            self.get_dij(X)
        if self.k == 'sign':
            d = self.dij - self.dc
            self.density = np.zeros(d.shape[0])
            for i, di in enumerate(d):
                self.density[i] = di[di < 0].size
        elif self.k == 'gauss':
            self.density = np.sum(np.exp(-self.dij ** 2 / (self.dc ** 2)),axis=1)
            # print(self.density)
        else:
            print('you need to input a kernel')
        return self.density

    def get_delta_i(self, X):
        if self.density is None:
            self.get_density_i(X)
        self.delta = np.zeros(self.density.shape[0])
        max_den = np.max(self.density)
        self.parent_index = np.zeros(self.density.shape[0]).astype('int')
        for i, di in enumerate(self.dij):
            if self.density[i] == max_den:
                self.delta[i] = np.max(di)
                self.parent_index[i] = i
            else:
                di[self.density <= self.density[i]] = float("inf")
                self.delta[i] = np.min(di)
                self.parent_index[i] = np.argmin(di)
        return self.delta, self.parent_index

    def dpc(self, X):
        self.dij = self.get_dij(X)
        self.density = self.get_density_i(X)
        self.delta, self.parent_index = self.get_delta_i(X)

    def draw_decision_graph(self, X):
        if self.delta is None:
            self.dpc(X)
        # print(self.density)
        # print(self.delta)
        plt.scatter(self.density, self.delta)
        plt.xlabel("density")
        plt.ylabel("delta")
        plt.show()

    def recall(self, i):
        p = self.parent_index[i]
        while self.cluster_list[p] == -1:
            p = self.parent_index[p]
        self.cluster_list[i] = self.cluster_list[p]

    def cluster(self, X, den, delt):
        if self.delta is None:
            self.dpc(X)
        self.thr_den = den
        self.thr_delt = delt
        n, _ = X.shape
        self.cluster_list = -1 * np.ones(n).astype('int')
        centers = np.where(np.logical_and(self.density > den, self.delta > delt))[0]
        print('centers:')
        print(len(centers))
        c_num = centers.shape[0]
        for i, center in enumerate(centers):
            self.cluster_list[center] = i
            self.parent_index[center] = center
        for i in range(n):
            self.recall(i)
        # print('cluster list:')
        # print(self.cluster_list)
        for i in range(c_num):
            x = X[self.cluster_list == i]
            plt.scatter(x[:, 0], x[:, 1])
            plt.scatter(X[centers[i],0],X[centers[i],1],c=color[-2],s=2)
        plt.title(f'k:{self.k},thr_delta:{self.thr_delt},dc:{self.dc}')
        plt.show()
        return self.cluster_list


def load_data(datapath):
    features = []
    with open(datapath, 'r') as f:
        for line in f.readlines():
            arr = []
            arr_str = line.strip().split(' ')
            for i in range(len(arr_str)):
                arr.append(float(arr_str[i]))
            features.append(arr)
    return np.array(features)


def test1():
    path1 = './Datasets/Aggregation.txt'
    data1 = load_data(path1)
    mydpc1 = DPC(16)
    mydpc1.draw_decision_graph(data1)
    clusterlist1 = mydpc1.cluster(data1, 0, 3)
    result1 = davies_bouldin_score(data1,clusterlist1)
    print('data1 cluster DBI:')
    print(result1)

def test2():
    path2 = './Datasets/D31.txt'
    data2 = load_data(path2)
    mydpc2 = DPC(1,'gauss')
    mydpc2.draw_decision_graph(data2)
    clusterlist2 = mydpc2.cluster(data2, 0, 2)
    result2 = davies_bouldin_score(data2, clusterlist2)
    print('data2 cluster DBI:')
    print(result2)

def test3():
    path3 = './Datasets/R15.txt'
    data3 = load_data(path3)
    mydpc3 = DPC(0.75,'gauss')
    mydpc3.draw_decision_graph(data3)
    clusterlist3 = mydpc3.cluster(data3, 0, 0.7)
    result3 = davies_bouldin_score(data3, clusterlist3)
    print('data3 cluster DBI:')
    print(result3)


if __name__ == '__main__':
    test1()
    test2()
    test3()
