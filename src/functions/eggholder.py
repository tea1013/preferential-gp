import numpy as np
from .function import Function
from scipydirect import minimize

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class EggHolder(Function):

    def __init__(self, noise_var, seed=1):
        self.dim = 2
        self.bounds = [[-512, 512], [-512, 512]]
        self.lengthscale_bound = [[0, 512]]
        super().__init__(self.dim, self.bounds, noise_var, seed)
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x']
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        x1 = x[0]
        x2 = x[1]

        res = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + 0.5 * x1 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

        return res

    def get_pool(self, K):
        K_ = int(np.sqrt(K))
        X = []
        for x1 in np.linspace(self.bounds[0][0], self.bounds[0][1], K_):
            for x2 in np.linspace(self.bounds[1][0], self.bounds[1][1], K_):
                X.append([x1, x2])

        return np.array(X)
