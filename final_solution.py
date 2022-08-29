# Import modules
import pandas as pd
import numpy as np
import os 

# Import NN classes
from layer_dense import *
from cost_functions import *
from optimisers import *

# Read csv
df = pd.read_csv('mnist_train.csv')

# Randomise df
df = df.sample(frac=1)

# Split df into training and testing data
df_test = df[0:8400]
df_train = df[8400:42000]

# Create training dataset
X_train = df_train.iloc[:,1:]
y_train = df_train.iloc[:, 0]

# Convert X into a numpy array and ramdomise the dataset
X_train = np.array(X_train)

# Initiate hidden layer with 784 input values and 10 neurons
hidden_layer = LayerDense(784, 10)
# Initiate ReLU activation object
relu = ActivationRelU()
# Initiate output layer with 10 input values and 10 neurons 
output_layer = LayerDense(10, 10)
# Initate softmax and cost functions with the ActivationSoftmaxCost object
softmax_cost = ActivationSoftmaxCost()

# Initiate optimiser object for back propagation 
sgd = Optimizer_SGD(learning_rate=0.001 ,decay=1e-4, momentum=100)

# Train in epochs. 501 iterations.
for epoch in range(501):
    # Forward propagation
    hidden_layer.forward(X_train)
    relu.forward(hidden_layer.output)
    output_layer.forward(relu.output)
    
    # Calculate error
    cost = softmax_cost.forward(output_layer.output, y_train)
    
    # Calculate accuracy from output of softmax and y
    predictions = np.argmax(softmax_cost.output, axis=1)
    accuracy = np.mean(predictions==y_train)
    
    # Print statistics per set of epochs
    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'err: {cost:.3f}, ' +
              f'lr: {sgd.current_learning_rate}')
        
    # Back propagation 
    softmax_cost.backward(softmax_cost.output, y_train)
    output_layer.backward(softmax_cost.dinputs)
    relu.backward(output_layer.dinputs)
    hidden_layer.backward(relu.dinputs)
    
    # Update weights and biases
    sgd.pre_update_params()
    sgd.update_params(hidden_layer)
    sgd.update_params(output_layer)
    sgd.post_update_params()
    
# Function to pass each sample in a dataset and output a results df.
def results(df): 
    # Initialise columns for results df
    image_id = []
    labels = []
    
    # Convert df into a numpy array
    data = np.array(df)
    
    # Create instance of the activation softmax class
    softmax = ActivationSoftmax()
    
    # Iterate throught df or data
    for index in range(len(df)):
        # Append indexx to ImageId columns 
        image_id.append(index)
        
        # Get sample row from data
        sample = data[index]
        
        # Pass sample to initiate forward propagation
        hidden_layer.forward(sample)
        relu.forward(hidden_layer.output)
        output_layer.forward(relu.output)
        softmax.forward(output_layer.output)
        
        # Append prediction to labels list
        labels.append(np.argmax(softmax.output))
    
    # Get path to output cv
    cwd = os.getcwd()
    path = cwd + "/submission"
    
    # Add columns list to a results dictionary 
    results = {'ImageId' : image_id,
               'Label' : labels}
    
    # Create new df and use results as the argument
    df_results = pd.DataFrame(results)
    
    # Write a csv file and output into the current directory
    df_results.to_csv(path,index = False)
    
# Read unseen data and call function
df = pd.read_csv('mnist_test.csv')
results(df)









































