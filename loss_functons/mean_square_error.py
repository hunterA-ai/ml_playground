import numpy as np
class mean_square_error:
    def __init__(self) -> None:
        pass

    def compute(self, y_true, y_pred):
        return np.sum((y_pred - y_true)**2)/len(y_true)
    
    def dC_dX(self, X, w, b, y):
        return np.outer(np.matmul(X, w) + b - y, w) # size(batch_size, n_in)

    def dC_dw(self, X, w, b, y):
        return 2 * np.sum(np.matmul(np.matmul(X, w) + b - y, X.T), axis=0) / X.shape[0] # size(batch_size, n_out)
    def dC_db(self, X, w, b, y):
        return 2 * np.sum(np.matmul(np.matmul(X, w) + b - y), axis=0) / X.shape[0]