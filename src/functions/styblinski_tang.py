import numpy as np
from .function import Function
from scipydirect import minimize

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class StyblinskiTang(Function):

    def __init__(self, noise_var, seed=1):
        self.dim = 2
        self.bounds = [[-5, 5], [-5, 5]]
        self.lengthscale_bound = [[0, 5]]
        super().__init__(self.dim, self.bounds, noise_var, seed)
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x']
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        res = 0
        for i in range(self.dim):
            xi = x[i]
            res += 0.5 * (xi**4 - 16 * xi**2 + 5 * xi)

        return res

    def get_pool(self, K):
        K_ = int(np.sqrt(K))
        X = []
        for x1 in np.linspace(self.bounds[0][0], self.bounds[0][1], K_):
            for x2 in np.linspace(self.bounds[1][0], self.bounds[1][1], K_):
                X.append([x1, x2])

        return np.array(X)
