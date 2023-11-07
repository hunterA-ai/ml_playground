from activation_functions.ReLU import ReLU
from loss_functions.mean_square_error import mean_square_error
from neural_networks.DenseLayer import DenseLayer
from neural_networks.OutLayer import OutLayer

class SimpleNeuralNetwork:
    """
    Represents a neural network as an array of {"DenseLayer", "OutLayer"} objects.
    The last element in the array must be of type "OutLayer"
    """
    def __init__(self, input_size, loss_func = mean_square_error()):
        self.nn_array = []
        self.input_size = input_size
        self.loss_func = loss_func


    def add_layer(self, num_neurons, activ_func=ReLU(), type="dense"):
        """
        type = {'dense', 'output'}

        New layer must have input size corresponding to previous layer's output size
        num_neurons - is the number of neurons in the current layer
        activ_func - is the activation function that should be applied to the outputs of this layer
        """
        num_in_n = 0
        if(len(self.nn_array) == 0):
            num_in_n = self.input_size
        else:
            num_in_n = self.nn_array[-1].W.shape[0]
        
        if(type == "output"):
            self.nn_array.append(OutLayer(
                input_size = num_in_n, 
                loss_func=self.loss_func))
        elif(type == "dense"):
            self.nn_array.append(DenseLayer(
                input_size=num_in_n,
                num_neurons=num_neurons,
                activ_func=activ_func
            ))
        else:
            raise(ValueError(f"Invalid Argument {type}, expected 'dense' or 'output'"))
        
        
    def describe_network(self):
        # weight matrix shape is (num_neurons, input_size)
        for layer in self.nn_array:
            print(layer)

    def forward_pass(self, input_matrix):
        for i in range(len(self.nn_array)):
            layer = self.nn_array[i]
            input_matrix = layer.batch_input(input_matrix)    
        return input_matrix
    
    def backward_pass(self, y_true, learning_rate):
        layer = self.nn_array[-1]
        dC_da = layer.backward(y_true, learning_rate)
        for i in range(len(self.nn_array)-1, 0, -1):
            layer = self.nn_array[i-1]
            dC_da = layer.backward(dC_da, learning_rate)