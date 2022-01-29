import numpy as np
from typing import Optional, Tuple
from typing_extensions import Literal
from matplotlib import pyplot as plt


class SVMClassifier(object):
    def __init__(self,
                 learning_rate: float,
                 max_iter: int,
                 C: float,
                 optimizer: Literal['GD', 'SMO', None] = None,
                 seed: Optional[int] = None):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.C = C
        self.optim = optimizer if optimizer else 'GD'
        self.seed = seed

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.train_data = (X, y)
        self.val_data = val_data
        np.random.seed(self.seed)
        X = self.__transfrom(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.err = {'train': [], 'val': []}
        if self.optim == 'GD':
            self.__SGDSolver(X, y)
        if self.optim == 'SMO':
            pass  # TO DO

    def predict(self, X: np.ndarray):
        X = self.__transfrom(X)
        pred = np.sign(np.dot(X, self.w) + self.b)
        pred[pred == 0] = 1
        return pred

    def score(self,
              X: np.ndarray,
              target: np.ndarray,
              metric: Literal['err', 'acc', 'f1'] = 'acc'):
        assert (X.shape[0] == target.size)
        if metric == 'acc' or 'err':
            y_pred = self.predict(X)
            acc = np.sum(y_pred == target) / target.size
            return acc if metric == 'acc' else 1 - acc
        if metric == 'f1':
            y_pred = self.predict(X)
            TP = np.sum(np.logical_and(y_pred == 1, target == 1))
            prec = TP / np.sum(y_pred == 1)
            recall = TP / np.sum(target == 1)
            return 2 * prec * recall / (prec + recall)

    def plot_boundary(self, X: np.ndarray, y: np.ndarray, sv: bool = True):
        assert X.shape[1] == 2
        x1, x2 = X[:, 0], X[:, 1]
        x1_lim = np.array([np.min(x1), np.max(x1)
                           ]) + np.array([-1, 1]) * .05 * np.ptp(x1)
        x2_lim = np.array([np.min(x2), np.max(x2)
                           ]) + np.array([-1, 1]) * .05 * np.ptp(x2)

        fig, ax = plt.subplots()
        ax.set_xlim(x1_lim[0], x1_lim[1])
        ax.set_ylim(x2_lim[0], x2_lim[1])
        acc = self.score(X, y, 'acc')
        ax.set_title("Boundary of SVM\naccuracy={}".format(round(acc, 3)))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        x1_sample = np.linspace(np.floor(x1_lim[0]),
                                np.ceil(x1_lim[1]),
                                num=100)

        def get_x2_sample(y=0):
            return (y - (self.b + self.w[0] * x1_sample)) / (self.w[1] + 1e-10)
        ax.plot(x1_sample, get_x2_sample(y=0), 'r-', linewidth=1.5, label='clf plane')
        ax.plot(x1_sample, get_x2_sample(y=1), 'r--', label='support plane')
        ax.plot(x1_sample, get_x2_sample(y=-1), 'r--')
        ax.scatter(x1[y == 1], x2[y == 1], c='#ff7f0e', label='positive')
        ax.scatter(x1[y == -1], x2[y == -1], c='#e377c2', label='negative')
        if sv:
            sv = (1 - (np.dot(X, self.w) + self.b) * y) >= 0
            ax.scatter(x1[sv], x2[sv], marker='o', s=150,
                       facecolors='none', edgecolors='#1f77b4',
                       linewidth=2, label='support vector')
        plt.legend(loc='upper left')
        plt.show()

    def plot_learning_curve(self):
        fig, ax = plt.subplots()
        ax.set_title('Learning curve with lr={}'.format(self.lr))
        ax.set_xlabel('epoch')
        ax.set_ylabel('error rate')
        ax.plot(np.arange(1, len(self.err['train']) + 1),
                self.err['train'],
                label='training error')
        if self.err['val']:
            ax.plot(np.arange(1, len(self.err['val']) + 1),
                    self.err['val'],
                    label='testing error')
        ax.legend()
        plt.show()

    def __transfrom(self, X: np.ndarray):
        return X

    def __update_err(self):
        self.err['train'].append(
            self.score(self.train_data[0], self.train_data[1], 'err'))
        if self.val_data:
            self.err['val'].append(
                self.score(self.val_data[0], self.val_data[1], 'err'))

    def __SGDSolver(self, X, y):
        for _loop in range(self.max_iter):
            e = 1 - (np.dot(X, self.w) + self.b) * y
            ei = (e >= 0)
            if not ei.any():
                break
            delta_w = self.w - self.C * np.dot(X[ei].T, y[ei])
            delta_b = -self.C * np.sum(y[ei])
            if np.sum(delta_w ** 2) + delta_b ** 2 < 1:
                break
            self.w -= self.lr * delta_w
            self.b -= self.lr * delta_b
            self.__update_err()
