import numpy as np
from scipy import linalg
class IterativeOutput(object):
    def __init__(self, x_shape, n_iter, detailed):
        self.detailed = detailed
        if detailed:
            self.iterations = np.empty((x_shape, n_iter))
            
    def store_iter(self, x, i):
        self.res = x
        if self.detailed:
            self.iterations[:,i] = x
        

def richardson(x0, A, b, m, omega, detailed=False):
    result = IterativeOutput(x0.shape[0], m, detailed)
    x = x0
    for i in range(m):
        x = x + omega*(b-A@x)
        result.store_iter(x, i)
    return result

def jacobi(x0, A, b, m, detailed=False):
    D = np.diag(A)
    R = A - np.diag(D)
    result = IterativeOutput(x0.shape[0], m, detailed)
    x = x0
    for i in range(m):
        x = (b - R.dot(x)) / D
        result.store_iter(x, i)
    return result

def gauss_seidel(x0, A, b, m, detailed=False):
    L = np.tril(A)
    U = A - L
    result = IterativeOutput(x0.shape[0], m, detailed)
    x = x0
    for i in range(m):
        x = linalg.solve_triangular(L, b - U.dot(x), lower=True)
        result.store_iter(x, i)
    return result