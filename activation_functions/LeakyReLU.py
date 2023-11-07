import numpy as np
class LeakyReLU:
    """
    Class instance of the Leaky Rectified Linear Unit function.
    Has method 'forward' that computes the Leaky ReLU for a matrix or vector along the row.
    """
    def __init__(self, leakiness):
        self.alpha = leakiness
    def forward(self, inputs):
        outp = np.maximum(self.alpha * inputs, inputs)
        return outp
    def derivative(self, inputs):
        outp = np.maximum(self.alpha, inputs)
        return outp