import numpy as np
import matplotlib.pyplot as plt
from .function import Function
from scipydirect import minimize
from sklearn.kernel_approximation import RBFSampler

class GPSamplePath(Function):
    def __init__(self, seed=1):
        self.dim = 1
        self.bounds = [[-3, 3]]
        self.y_bounds = [-2, 2]
        super().__init__(self.dim, self.bounds, seed)
        self.fit()
        self.min, self.max = self.get_min_max()
        res = minimize(self.value_std, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x'][0]
        self.y_opt = -self.value_std(self.x_opt)

    def base_function(self, x):
        res = (6 * x - 2)**2 * np.sin(12 * x - 4)

        return res

    def fit(self):
        X = np.linspace(self.bounds[0][0], self.bounds[0][1], 3)
        Y = np.random.uniform(self.y_bounds[0], self.y_bounds[1], 3)
        X = X.reshape(-1, 1)
        self.rbf_feature = RBFSampler(gamma=1, n_components=30)
        self.rbf_feature.fit(np.atleast_2d(X[0]))
        phi_X = self.rbf_feature.transform(X)
        self.w = np.linalg.inv(phi_X.T.dot(phi_X)).dot(phi_X.T).dot(Y)

    def get_min_max(self):
        X = np.linspace(self.bounds[0][0], self.bounds[0][1], 10000)
        Y = self.value(X)
        return np.min(Y), np.max(Y)

    def value(self, x):
        x = x.reshape(-1, 1)
        res = self.rbf_feature.transform(x).dot(self.w)
        return res

    def value_std(self, x):
        res = self.value(x)
        res = (res - self.min) / (self.max - self.min)

        return res

    def get_pool(self, K):
        return np.linspace(self.bounds[0][0], self.bounds[0][1], K)

    def plot(self):
        x_range = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
        y = self.value_std(x_range)
        plt.plot(x_range, y)
        plt.show()

