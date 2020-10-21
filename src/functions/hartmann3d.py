import numpy as np
from .function import Function
from scipydirect import minimize

class Hartmann3d(Function):

    def __init__(self, noise_var, seed=1):
        self.dim = 3
        self.bounds = [[0, 1], [0, 1], [0, 1]]
        self.lengthscale_bound = [[0, 1]]
        super().__init__(self.dim, self.bounds, noise_var, seed)
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x']
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]
                      ])
        P = 1e-4 * np.array([[3689, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381, 5743, 8828]
                             ])

        res = 0
        for i in range(4):
            sum_exp_in = 0
            for j in range(3):
                sum_exp_in += A[i][j] * (x[j] - P[i][j])**2
            res += alpha[i] * np.exp(-sum_exp_in)

        return res
