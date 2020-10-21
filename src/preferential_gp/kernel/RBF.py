import numpy as np
import GPy

class RBF:
    def __init__(self, input_dim, lengthscale, variance=1):
        self.input_dim = input_dim
        self.lengthscale = lengthscale
        self.variance = variance
        self.k = GPy.kern.RBF(input_dim, lengthscale=lengthscale, variance=variance)

    def __call__(self, x_i, x_j=None):
        return self.k.K(x_i, x_j)

    def set(self, lengthscale):
        self.lengthscale = lengthscale
        self.k = GPy.kern.RBF(self.input_dim, lengthscale=lengthscale, variance=self.variance)
