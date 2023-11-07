import numpy as np


class DenseLayer:
    """
    Represents a dense layer of neurons, as a weight matrix (W) and bias (b), plus an activation function. Allows for forward and backward passes, which compute the output
    and gradient, respectively. Backward implementation is intended for batch gradient descent.
    """
    def __init__(self, input_size, num_neurons, activ_func):
        self.num_in_n = input_size
        self.num_out_n = num_neurons
        # self.weight_matrix = np.array([np.random.rand(input_size) for _ in range(num_neurons)])
        # Above line has been upgraded to line below
        self.W = np.random.randn(self.num_out_n, self.num_in_n)/2
        self.b = np.random.randn(self.num_out_n)/2
        self.activation_func = activ_func

    def batch_input(self, X):
        """
        Returns the matrix product [input_matrix] * [weight_matrix]^T of dimensions
        (batch_size, num_in_neurons) * (num_in_neurons, num_out_neurons) = (batch_size, num_out_neurons)

        
        XW^T + b is (batch_size, num_out_neurons) + (num_out_neurons), where the bias is brodcast for each row
        """    
        self.X = X
        self.batch_size = X.shape[0] 
        self.raw_output = np.dot(self.X, self.W.T) + self.b
        self.activation_output = self.activation_func.forward(self.raw_output)
        return self.activation_output
    
    
    def backward(self, error_matrix, learning_rate):
        """
        Given the error vector dC/da^(l), returns the new error vector for the next layer, dC/da^(l-1)

        C = cost func
        a^(l) = activation function at layer l
        z = XW^T + b
        """
        eta = learning_rate

        dC_da_1 = error_matrix # (batch_size, out_n)
        da_dz = self.activation_func.derivative(self.raw_output) # (batch_size, out_n)
        dC_dz = dC_da_1 * da_dz # (batch_size, num_out_n)

        # Error Gradient
        dC_dX = np.tensordot(dC_dz, self.W, axes=(1,0)) # (batch_size, in_n)
        # Gradient of W (average weight at w)
        dC_dw = np.sum(np.matmul(dC_dz.T , self.X), axis=0) / self.batch_size # (out_n)
    
        # Gradient of b
        dC_db = np.sum(dC_dz, axis=0) / self.batch_size # (out_n)

        self.W = self.W - (eta * dC_dw)
        self.b = self.b - (eta * dC_db)
        return dC_dX
    