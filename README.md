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

## The Mathematics of Backpropagation

### Forward Pass
Let our output for a particular layer be $$z\coloneqq X W^T + b$$
where $b$ is added to each row/instance via broadcasting  
- $X \in \mathbb{R}^{(\text{batchSize, inputNeurons})}$ is the batch data input for a layer.
- $W^T \in \mathbb{R}^{(\text{inputNeurons, outputNeurons})}$ is the weight matrix for a layer.
- $XW^T \in \mathbb{R}^{(\text{batchSize, outputNeurons})}$ is the linear transformation in a layer.
- $b \in \mathbb{R}^{\text{outputNeurons}}$ is the bias vector.
- Let $a(z)$ be the activation function, then $a(z) \in \mathbb{R}^{(\text{batchSize, inputNeurons})}$

Additionally, let $X_{ij}$ represent the element of $X$ in the $i^{th}$ row and $j^{th}$ column. $X_{i:}$ will represent row $i$ and $X_{:j}$ will represent column $j$.  
### Backward Pass
#### Out Layer (MSE loss)
*Error gradient to be passed backwards*  
Let $C: \mathbb{R}^n \to \mathbb{R}$ be our cost function which is MSE $((Xw^T + b)-y)^2$. The cross product below may look unintuitive at first, but consider that we must propogate the error to all neurons in the previous layer, where the number of neurons in the previous layer is the dimension of $w^T$.  
- $$\frac{\partial C}{\partial X} = 2 * ((Xw^T + b)-y) \otimes w \qquad shape (\text{batchSize, inputNeurons}) \tag{1.1}$$  

*Error gradient for weight update*  
The following are the partial derivatives for our weight updates of $w$ and $b$.  
- $$\frac{\partial C}{\partial b} = 2 * ((Xw^T + b) - y) \tag{1.2}$$
- $$\frac{\partial C}{\partial w} = 2 * ((Xw^T + b)-y) X  \tag{1.3}$$

Here is what the above function looks like in the python code  `2 * np.sum(np.dot((np.dot(X, w) + b - y), X)) / X.shape[0]`. Notice that we have to normalize by `X.shape[0]` which is `batchSize`. We are simply doing a batch update by averaging the weight gradient.



#### Regular Layer
*Error gradient to be passed backwards*  
Let $l-1$ represent the current layer in a neural network. The following notation is very tricky as we have two cost functions, one at the current layer $C_{l-1}$ which we are computing, and one at the next layer $C_{l}$ which is already computed. For simplicity we will drop the layer subscript, and adhere to the following guidelines. $C$ on the LHS of the equation is the cost at the current layer and $C$ on the RHS of the equation is the cost at the next layer. To further simplify notation, we will use $C'$ and $a'$ to denote the derivatives. The derivative is taken with respect to the varaible defined on the left, it will look like $\frac{\partial C}{\partial W_{i}} = 
C'(a(...)) * a'(...) * ...$ i.e. in this case $C'$ and $a'$ are shorthand for $\frac{\partial C}{\partial W_{i}}$ and $\frac{\partial a}{\partial W_{i}}$, respectively.

We simplify the computation of $\frac{\partial C}{\partial X}$ by first computing the error for the $i^{th}$ element of the batch and the $j^{th}$ neuron, $\frac{\partial C}{\partial X_{ij}}$. This computation involves taking the sum of errors over all output neurons $k$. In other words, the sum of the errors flowing backward to neuron $j$.  
- $$\frac{\partial C}{\partial X_{ij}} = \sum_{k=1}^{\text{outputNeurons}} \left[ C'(a(X_{ij}(W^T)_{jk}+b_k)) * a'(X_{ij}W^T_{jk} + b_k) * W^T_{jk} \right] \tag{1.4}$$  
Where $\frac{\partial C}{\partial X_{ij}}$ has `shape(1,)`. The above notation abuses the fact that $C$ can take a vector of any size and in this case it is a vector of dimension 1. Hopefully the above equation illuminates some of the hidden operations in the following:
- $$\frac{\partial C}{\partial X} = \sum_{k=1}^{\text{outputNeurons}} \left[ C'(a(X W^T_{:k} + b_k)) * a'(X W^T_{:k} + b_k) \otimes  W^T_{:k}\right]  \tag{1.5}$$
Where $\frac{\partial C}{\partial X}$ has `shape(batchSize, inputNeurons)`  

The computation for $\frac{\partial C}{\partial X}$ looks something like `np.tensordot(C'(a(z)) * a'(z), W, axes=(1,0))`

Again lets ellaborate on the tensor operation. We take the dot product of the rows of $(C'* a')$ with the columns of $W$, i.e. the dot product of two vectors which have length `outputNeurons`.


*Error gradient for weight update*
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

