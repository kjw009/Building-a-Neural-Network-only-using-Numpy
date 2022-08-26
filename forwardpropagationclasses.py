# Import numpy for array handling
import numpy as np

# Create a neural network layer of n_neurons using n_inputs.
class LayerDense:
    # Initialise weights and biases
    def __init__(self, n_inputs,n_neurons):       
        self.weights = 0.01 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))
    
    # Method to caluclate the output of the layer object
    def forward(self, inputs):
        # Calculate output values from the inputs, weights and biases
        self.output = np.dot(inputs, self.weights.T) + self.biases

# Rectified linear activation class
class ActivationReLU:
    def activate(self, inputs):
        # If input is greater than 0 return input else return 0
        self.output = np.maximum(0, inputs)

# Softmax activation class
class ActivationSoftmax:
    def activate(self, inputs):
        # Exponentiate input values. The inputs will be subrstracted by the max value of inputs to prevent overflow.
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
        # Normalise the values
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities