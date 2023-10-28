import numpy as np
#only single variable functions accepted
class mean_square_error:
    def __init__(self) -> None:
        pass

    def compute(self, y_true, y_pred):
        return np.sum(np.abs((y_pred - y_true)))/len(y_true)
    
    def derivative(self, y_true, y_pred):
        return np.sum(np.sign((y_pred-y_true)))/len(y_true)