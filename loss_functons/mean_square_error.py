import numpy as np
class mean_square_error:
    def __init__(self) -> None:
        pass

    def compute(self, y_true, y_pred):
        return np.sum((y_pred - y_true)**2)/len(y_true)
    
    def derivative(self, y_true, y_pred):
        return 2 * np.sum((y_pred-y_true))/len(y_true)
