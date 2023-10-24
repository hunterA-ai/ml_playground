import numpy as np
class SoftMax:
    """
    Class instance of the SoftMax function. Has method 'forward' that computes the SoftMax score for a matrix or vector along the row
    """
    def __init__(self):
        pass
    def forward(self, matrix):
        exp_values = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        normalization_constant = np.sum(exp_values, axis=1, keepdims=True)
        probability_vector = exp_values / normalization_constant
        return probability_vector
