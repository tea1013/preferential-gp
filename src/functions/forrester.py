import numpy as np
from .function import Function
from scipydirect import minimize

class Forrester(Function):

    def __init__(self, noise_var, seed=1):
        self.dim = 1
        self.bounds = [[0, 1]]
        self.lengthscale_bound = [[0, 1]]
        super().__init__(self.dim, self.bounds, noise_var, seed)
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x'][0]
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        res = (6 * x - 2)**2 * np.sin(12 * x - 4)
        return res

    def get_pool(self, K):
        return np.linspace(self.bounds[0][0], self.bounds[0][1], K)
