import numpy as np

def diaconis_shahshahani(N):
    theta_0 = np.random.uniform(0, np.pi)
    matrix = np.array([[np.cos(theta_0), -np.sin(theta_0)], [np.sin(theta_0), np.cos(theta_0)]])
    for i in range(2, N):
        # generate a vector on the n-sphere
        v = np.random.normal(size=(i+1, 1))
        v = v / np.linalg.norm(v)
        
        # embed in the next largest matrix
        y = np.zeros((i+1, 1))
        y[-1] = 1.0
        matrix = np.r_[np.c_[matrix, y[:-1]], y.T]
        
        # rotate the matrix so the last column is v
        
        z = y - v.T.dot(y)*v
        z = z / np.linalg.norm(z)
        
        cost = y.T.dot(v)
        sint = np.sqrt(1-cost*cost)
        
        tmp = np.column_stack([v, z])
        mat = np.c_[np.r_[cost, -sint], np.r_[sint, cost]]
        R = np.eye(i+1)-v.dot(v.T)-z.dot(z.T) + tmp.dot(mat).dot(tmp.T)
        matrix = R.dot(matrix)
    return matrix
