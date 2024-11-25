## Introduction
This project is about creating basic neural network using only linear algebra and object oriented programming. The mathematics behind the forward and backward (training) operations, as well as an example regression task, is provided as a proof of correctness. The code is in Python and the only package used is numpy, which speeds up the vector/matrix/tensor operations. The purpose of this project is simply as an exercise, similar to what may be found in a course on deep learning. The goal of this exercise is to gain a deeper familiarity with neural networks & the machine learning landscape in general. This work is self-guided and self-motivated.


## Code Structure
1. *loss_functions*
	1. *loss_functions/mean_square_error.py* - This function has the capability to compute the loss of a regression problem. Also determines the derivative with respect to each variable in the input $(X,w,b)$, the expected input form is $Xw^T + b$. This allows for back-propogation & training of the neural network.
2. *activation_functions*
	1. *activation_functions/ReLU.py* Implementation of the rectified linear function, derivative included.
	2. *activation_functions/LeakyReLU.py* Implementation of a leaky rectified linear function, derivative included.
3. *neural_networks*
	1.  *neural_networks/SimpleNeuralNetwork.py* Represents a simple feed forward neural network capable of being trained on regression problems. The number and size of layers is adjustable. Training is implemented through batch gradient descent
	2. *neural_networks/DenseLayer.py* Represents a dense layer of neurons as a weight matrix (W) and bias (b). Includes forward and backward functions for training and prediction.
	3. *neural_networks/OutLayer.py* Represents the output layer of neurons as a weight vector (w) and bias (b), with functions to compute the single variable output, loss of the network, and error gradient from the loss.

## Math
### Forward Pass
$$z\coloneqq X W^T + b$$
where $b$ is added to each row/instance via broadcasting
- $X \in \mathbb{R}^{(\text{batchSize, inputNeurons})}$
- $W^T \in \mathbb{R}^{(\text{inputNeurons, outputNeurons})}$
- $XW^T \in \mathbb{R}^{(\text{batchSize, outputNeurons})}$
- $b \in \mathbb{R}^{\text{outputNeurons}}$
- Let $a(z)$ be the activation function, then $a(z) \in \mathbb{R}^{(\text{batchSize, inputNeurons})}$

### Backward Pass
#### Out Layer (MSE loss)
*Error gradient to be passed backwards*  
- $$\frac{\partial C}{\partial X} = 2 * ((Xw^T + b)-y) \otimes w \qquad shape (\text{batchSize, inputNeurons})$$  

*Error gradient to update weights*  
- $$\frac{\partial C}{\partial w_j} = 2 * ((Xw^T + b) - y) 
\cdot (X_{1j}, \dots, X_{nj}) $$
- $$\frac{\partial C}{\partial w} = 2 * ((Xw^T + b)-y) X = 2 * np.dot(((Xw^T + b)-y) ,X)$$



#### Regular Layer
*Error gradient to be passed backwards*  
For the $j^{th}$ neuron of instance $i$, we take the sum of errors over all outputs $k$. In other words, the sum of the errors flowing backward to neuron $j$.  
- $$\frac{\partial C}{\partial X_{ij}} = \frac{\partial C}{\partial x_j^{(i)}} = \sum_{k=1}^{\text{outputNeurons}} \left[ \frac{\partial C_k}{\partial a_k} \cdot a_k'\left(x^{(i)}_j W^{T} + b\right) \cdot W_{ki} \right]$$  
- $$\frac{\partial C}{\partial X} = \sum_{k=1}^{\text{outputNeurons}} \left[ C_k'(a) *a_k'(X W^{T} + b) \otimes  w_{k}\right] \qquad shape(\text{batchSize, inputNeurons})$$
- $$\frac{\partial C}{\partial X} = np.tensordot\left( C'(a(z)) * a'(z), W ,axes=(1,0)\right) \qquad shape(\text{batchSize, inputNeurons})$$  

Above is taking the dot product of the rows of $(C'* a')$ with the columns of $W$, i.e. the dot product of two dimension `outputNeurons` vectors.

*Error gradient to update weights*
$$\frac{\partial C}{\partial W_{i}} = 
C'(a(Xw_i + b_i)) * a'(Xw_i + b_i) *  \begin{pmatrix} 
x_i^{(1)} \\
\vdots \\
x_i^{(batchSize)}
\end{pmatrix}$$
Taking $\partial W_i$ is the derivative with respect to the $i^{th}$ output neuron. Notice we only need to take the derivative wrt the output neurons, since each weight $w_{ij}$ is updated entirely according to the output neuron $i$.
$$\frac{\partial C}{\partial W} =( C'(a(XW^T + b)) * a'(XW^T+b)) X^T$$
Average weight update along the batch
$$\nabla_w C = np.sum\left((C'(a(XW^T + b)) * a'(XW^T+b))X^T, axis=0\right) \div (\text{batchSize}) \qquad shape(outputNeurons)$$
Update weights
$$ W = W - \eta \nabla_w C$$


## Application
found in *notebooks/nn_test.ipynb*


## References

Géron, Aurélien. _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O'Reilly, 2019.

NumPy. (2023, September 16). _NumPy v1.26 Manual_. [https://numpy.org/doc/1.26/](https://numpy.org/doc/1.26/)

