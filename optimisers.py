# SGD optimiser class
class Optimizer_SGD:
    # Initialise optimiser and set default params
    def __init__(self, learning_rate = 1.0, decay = 0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay 
        self.iterations = 0
        self.momentum = momentum
    
    # Call once before any parameter updates
    def pre_update_params(self):
        # If decay argument, apply decay to current learning rate calculations
        self.current_learning_rate = self.learning_rate * \
                                     (1.0 / (1.0 + self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):
        # If momentum was passed
        if self.momentum:
            # If layer output does not contain momentum arrays
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
                weight_updates = \
                                 self.momentum * layer.weight_momentums - \
                                 self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates
                
                # Update bias
                bias_updates = self.momentum * layer.bias_momentums - \
                               self.current_learning_rate * layer.dbiases
                layer.bias_momentums = bias_updates
                
                # If no momentum is used 
            else:
                weight_updates = -self.current_learning_rate * \
                                  layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases
                
            # Update weights and biases
            layer.weights += weight_updates
            layer.biases += bias_updates
        
    # Call once after a params update
    def post_update_params(self):
        self.iterations += 1             