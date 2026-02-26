
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
        # insert constant ones for bias
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # gradient descent for n_iters
        for _ in range(self.n_iters):
            y_pred = X.dot(self.w)

            # calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred))**2 + self.regularization(self.w)
            self.training_errors.append(mse)

            # gradient of l2 loss and update weights
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

class LinearRegression(Regressor):
    def __init__(self, n_iters=100, learning_rate=0.001, gradient_descent=True):
        self.gradeinet_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(
            n_iters = n_iters,
            learning_rate = learning_rate
        )
    def fit(self, X, y):
        # if not gradient descent => least square approximation of w

        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_in = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_in.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

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