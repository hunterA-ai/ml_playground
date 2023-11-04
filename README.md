# ml_playground
Playing around with various machine learning related tasks

Code and Math Proof of Correctness

#### Initialization
```
class nn_layer:
	def __init__(...):
		self.num_in_n
		self.num_out_n
		self.batch_size
```


#### Forward Pass
```
	def batch_input(input_matrix):
		self.input_matrix
		self.raw_output = np.matmul(self.input_matrix, self.weight_matrix.T)
		self.activation_output = self.activation_func.forward(self.raw_output)
		return self.activation_output
```

- $z\coloneqq X W^T + \begin{pmatrix} b \\ b \\ \vdots \\b \end{pmatrix}$
Where $b$ is added to each row, componentwise
- $X \in \mathbb{R}^{(batch\_size, num\_in\_n)}$
- $W^T \in \mathbb{R}^{(num\_in\_n, num\_out\_n)}$
- $XW^T \in \mathbb{R}^{(batch\_size, num\_out\_n)}$
- $b \in \mathbb{R}^{num\_out\_n}$
- Let $a(z)$ be the activation function, then $a(z) \in \mathbb{R}^{(batch\_size,num\_in\_n)}$

