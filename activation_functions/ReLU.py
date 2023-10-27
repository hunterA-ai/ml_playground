import numpy as np
class ReLU:
    """
    Class instance of the Rectified Linear Unit function.
    Has method 'forward' that computes the ReLU for a matrix or vector along the row.
    """
    def __init__(self):
        pass
    def forward(self, inputs):
        outp = np.maximum(0, inputs)
        return outp
    def derivative(self, inputs):
        outp = np.maximum(0, inputs)
        return outp