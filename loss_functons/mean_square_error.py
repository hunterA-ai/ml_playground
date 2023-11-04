import numpy as np
class mean_square_error:
    def __init__(self) -> None:
        pass

    def compute(self, y_true, y_pred):
        return np.sum((y_pred - y_true)**2)/len(y_true)
    
    def dC_dX(self, X, w, b, y):
        print("Xw +b shape", (np.dot(X, w) + b).shape)
        print("Xw +b shape", (np.dot(X, w) + b - y).shape)
        return np.outer((np.dot(X, w) + b) - y, w) # shape(batch_size, n_in)

    def dC_dw(self, X, w, b, y):
        return 2 * np.sum(np.dot((np.dot(X, w) + b - y), X)) / X.shape[0] # shape(1)
    
    def dC_db(self, X, w, b, y):
        return 2 * np.sum(np.dot(X, w) + b - y) / X.shape[0]