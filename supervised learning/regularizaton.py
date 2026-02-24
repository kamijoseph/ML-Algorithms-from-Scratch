
# lasso, ridge and elastic net regularization
import numpy as np

# lasso(l1) regularization
class LassoRegularization():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)

# l2 regularization
class RidgeRegularization():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w

class ElasticNet():
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

    def grad(self, w):
        pass
