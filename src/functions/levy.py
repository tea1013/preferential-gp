import numpy as np
from .function import Function
from scipydirect import minimize

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class Levy(Function):

    def __init__(self, noise_var, seed=1):
        self.dim = 2
        self.bounds = [[-10, 10], [-10, 10]]
        self.lengthscale_bound = [[0, 10]]
        super().__init__(self.dim, self.bounds, noise_var, seed)
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x']
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        res = 0
        w_1 = 1 + (x[0] - 1) / 4.0
        w_2 = 1 + (x[1] - 1) / 4.0
        res += np.sin(np.pi * w_1)**2
        for i in range(self.dim - 1):
            w_i = 1 + (x[i] - 1) / 4.0
            res += (w_i - 1)**2 * (1 + 10 * np.sin(np.pi * w_i + 1)**2) + (w_2 - 1)**2 * (1 + np.sin(2 * np.pi * w_2)**2)
            # 実装

        return res

    def get_pool(self, K):
        K_ = int(np.sqrt(K))
        X = []
        for x1 in np.linspace(self.bounds[0][0], self.bounds[0][1], K_):
            for x2 in np.linspace(self.bounds[1][0], self.bounds[1][1], K_):
                X.append([x1, x2])

        return np.array(X)

