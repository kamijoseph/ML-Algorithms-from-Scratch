
# linear regularization
import numpy as np
import math
class Regressor(object):
    def __init__(self, n_iters, learning_rate):
        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        # initializing weights randomly [-1/N, 1/N]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class LinearRegression(Regressor):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        pass

class LassoRegression(Regressor):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class PolynomialRegression(Regressor):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class RidgeRegression(Regressor):
    def __init__(self):
        super().__init__()
        pass

class PolynomialRidgeRegression(Regressor):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class ElasticNet(Regressor):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass