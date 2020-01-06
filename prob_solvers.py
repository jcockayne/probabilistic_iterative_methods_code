import numpy as np
from scipy import linalg
class ProbabilisticIterativeOutput(object):
    def __init__(self, x_shape, n_iter, detailed):
        self.detailed = detailed
        if detailed:
            self.means = np.empty((x_shape, n_iter))
            self.covs = np.empty((x_shape, x_shape, n_iter))
            
    @property
    def iterations(self):
        return self.mean
    
    def store_iter(self, x, Sigma, i):
        self.mean = x
        self.cov = Sigma
        if self.detailed:
            self.means[:,i] = x
            self.covs[:,:,i] = Sigma
            
def richardson(x0, Sigma0, A, b, m, omega, detailed=False):
    result = ProbabilisticIterativeOutput(x0.shape[0], m, detailed)
    x = x0
    Sigma = Sigma0
    G = np.eye(x0.shape[0]) - omega*A
    for i in range(m):
        x = x + omega*(b - A@x)
        Sigma = G @ Sigma @ G.T
        result.store_iter(x, Sigma, i)
    return result

def jacobi(x0, Sigma0, A, b, m, detailed=False):
    result = ProbabilisticIterativeOutput(x0.shape[0], m, detailed)
    D = np.diag(A)
    Dinv = np.diag(1./D)
    R = A - np.diag(D)
    G = (Dinv @ R)
    x = x0
    Sigma = Sigma0
    
    for i in range(m):
        x = (b - R.dot(x)) / D
        Sigma = G @ Sigma @ G.T
        result.store_iter(x, Sigma, i)
    return result

def gauss_seidel(x0, Sigma0, A, b, m, detailed=False):
    result = ProbabilisticIterativeOutput(x0.shape[0], m, detailed)
    L = np.tril(A)
    U = A - L
    G = linalg.solve_triangular(L, U, lower=True)
    x = x0
    Sigma = Sigma0
    
    for i in range(m):
        x = linalg.solve_triangular(L, b - U.dot(x), lower=True)
        Sigma = G @ Sigma @ G.T
        result.store_iter(x, Sigma, i)
    return result