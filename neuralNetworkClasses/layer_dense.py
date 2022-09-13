import numpy as np

# Layer class
class LayerDense:
    # Initialise object
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        
# Forward pass. Renamed from calculate as this is the forward propagation step
    def forward(self, inputs):
        # We will need the inputs to calculate the partial derivative with respects to weights during backcprogation.
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
 # Backward step. The gradient of the subsequent layer with respects to inputs is used as the paramenter
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
