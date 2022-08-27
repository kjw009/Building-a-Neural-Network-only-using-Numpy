# RelU activation class
class ActivationRelU:
    # Forward step
    def forward(self, inputs):
# We need the inputs for the backward step to calculate the Relu activation function gradient
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    # Backward step
    def backward(self, dvalues):
        # Make a copy
        self.dinputs = dvalues.copy()
        # If the input values were negative, the value or the gradient is zero.
        self.dinputs[self.inputs <= 0] = 0

# Combine softmax and cost function for faster backward propagation step
class ActivationSoftmaxCost():
    # Creates activation and cost function objects
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.cost = CategoricalCrossEntropy()
        
    # Forward step
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return error
        return self.cost.calculate(self.output, y_true)
    
    # Backward step. Takes in the softmax output. (Not a derivitve)
    def backward(self, dvalues, y_true):
        # Number of samples 
        samples = len(dvalues)
        
        # Ensure labels are scalar values
        if len(y_true.shape) == 2:
        # Return a vector of index where the one_hot_coded vector is positive
            y_true = np.argmax(y_true, axis = 1)
            
        # Copy the derivative values from subsequent layer function
        self.dinputs = dvalues.copy()
        # Calculate the gradient . Chain rule
        self.dinputs[range(samples), y_true] -= 1
        # Normalise the gradient
        self.dinputs = self.dinputs / samples