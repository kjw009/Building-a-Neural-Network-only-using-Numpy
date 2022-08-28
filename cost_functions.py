# Import numpy for array handling
import numpy as np

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
        
# Class to calculate error of an output
class Cost:
# Takes in the output layer values and the correct labels for the samples
    def calculate(self, output, y):
        # Calculate the losses for the corresponding sample in the set
        costs = self.forward(output, y)
        # Calculate mean loss in the sample
        mean_cost = np.mean(costs)
        # Return loss
        return mean_cost

# Catergorical cross entropy class that will inherit the base Cost class
class CategoricalCrossEntropy(Cost):
# Calculate the error using the predicted values from the output and 
# its corrresponding labels
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
         # Clip the predicted values to prevent error caused by inf
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # If the y_true are scalar values else use one-hot encoded labels
        if len(y_true.shape) == 1:
        # Retrieve the highest probability from each sample output from y_pred
            confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate negative log of the confidences and return the value
        negative_log_confidences = -np.log(confidences)
        return negative_log_confidences
    
    # Backward step. Takes in the softmax output as dvalues. 
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in the samples
        labels = len(dvalues[0])
        
        # If labels are scalar values, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalise gradient
        self.dinputs = self.dinputs / samples
        
# Softmax activation class
class ActivationSoftmax:
    def forward(self, inputs):
        self.inputs = inputs
        # Exponentiate input values. The inputs will be subrstracted by the max value of inputs to prevent overflow.
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
        # Normalise the values
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
    
    # Backward step. Takes in the gradient/derivitive of the subsequent layer
    def backward(self, dvalues):
        # Create an empty array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradient
        for index, (single_output, single_dvalues) in \
        enumerate(zip(self.output, dvalues)):
            
            # Reshape sample softmax output into a column vector
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the sample output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            
        # Calculate gradient (Chain Rule) and append dinputs.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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