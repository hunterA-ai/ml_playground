import numpy as np

class OutLayer:
    """
    Represents the output layer of a neural network as a weight vector (w) with shape(in_n) and scalar bias (b)
    """
    def __init__(self, input_size, loss_func):
        self.num_in_n = input_size
        self.loss_func = loss_func
        self.num_out_n = 1
        self.W = np.random.randn(self.num_in_n)/2
        self.b = np.random.randn(self.num_out_n)/2

    def batch_input(self, X):
        self.X = X
        self.batch_size = X.shape[0]
        """
        Returns the matrix product [input_matrix] * [weight_matrix]^T of dimensions
        (batch_size, num_in_neurons) * (num_in_neurons, num_out_neurons) = (batch_size, num_out_neurons)


        XW^T + bias is (batch_size, num_out_neurons) + (num_out_neurons), where the bias is brodcast for each row
        """     
        self.raw_output = np.dot(self.X, self.W.T) + self.b
        return self.raw_output
    
    
    def backward(self, y_true, learning_rate):
        """
        Given the error vector dC/da^(l), returns the new error vector for the next layer, dC/da^(l-1)

        C = cost func
        a^(l) = activation function at layer l
        z = XW^T + b
        """
        eta = learning_rate

        # Error Gradient
        dC_dX = self.loss_func.dC_dX(X=self.X, w=self.W, b=self.b, y=y_true)
        # Gradient of W
        dC_dw = self.loss_func.dC_dw(X=self.X, w=self.W, b=self.b, y=y_true)
        # Gradient of b
        dC_db = self.loss_func.dC_db(X=self.X, w=self.W, b=self.b, y=y_true)
        self.W = self.W - (eta * dC_dw)
        self.b = self.b - (eta * dC_db)
        return dC_dX
    