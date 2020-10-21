import numpy as np
from .function import Function
from scipydirect import minimize

class Hartmann4d(Function):

    def __init__(self, noise_var, seed=1):
        self.dim = 4
        self.bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
        self.lengthscale_bound = [[0, 1]]
        super().__init__(self.dim, self.bounds, noise_var, seed)
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x']
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]
                      ])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]
                             ])

        res = 0
        for i in range(4):
            sum_exp_in = 0
            for j in range(4):
                sum_exp_in += A[i][j] * (x[j] - P[i][j])**2
            res += alpha[i] * np.exp(-sum_exp_in)

        res = (1.1 - res) / 0.839

        return res

