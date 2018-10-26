#
#
# Deep Neural Network implementation
#
#

import numpy as np

class Network:
    
    # Net initialization
    def __init__(self, layer_sizes):
        # Neuron activations
        # N arrays of size n_L+1 x 1
        self.As = tuple([np.concatenate((np.zeros((y,1),dtype=np.float32),[[1]]),axis=0) for y in layer_sizes])

        # Neuron z values
        # N-1 arrays of size n_L+1 x 1 (skipping first layer)
        #Zs = tuple([np.concatenate((np.zeros((y,1),dtype=np.float32),[[1]]),axis=0) for y in layer_sizes])
        self.Zs = tuple([np.zeros((y,1),dtype=np.float32) for y in layer_sizes[1:]]) # DIFF
        
        # Weights
        # N-1 matrices of size  n_L x n_Lm1+1 (skipping first layer)
        # Initialized as random values (standard normal distribution)
        self.weights = tuple([np.random.randn(y,x+1) for x,y in zip(layer_sizes[:-1],layer_sizes[1:])])

        # Activation functions to use
        # TODO: Specify in constructor
        self.activation_function = sigmoid
        self.activation_prime_function = sigmoid_prime
        self.cost_function = cost
        self.cost_prime_function = cost_prime

        # Helful variables
        self.L = len(layer_sizes)
    
    # Forward propagate network
    # X: Input data array size of first layer
    # Return value: Calculated last layer
    def feedforward(self, X):
        # Set first layer to input
        # Force convert list or array to single column
        # [True] replaces every element but keeps the original object
        self.As[0][:-1] = np.transpose(np.matrix(X).flatten())
        
        # Go through every layer and calculate neuron activation
        for i in range(self.L-1):
            # Calculate z values for next layer
            self.Zs[i][True] = np.dot(self.weights[i], self.As[i]) # DIFF
            # Calculate activation for next layer
            self.As[i+1][:-1] = self.activation_function(self.Zs[i])

        # Return last layer
        return self.As[-1][:-1]

    # Backpropagate network for single training sample 
    # X: training input
    # Y: training label
    # Return value: cost gradient
    def backpropagate(self, X, Y):
        # Force convert list or array to single column
        X = np.transpose(np.matrix(X).flatten())
        Y = np.transpose(np.matrix(Y).flatten())
        
        # Feedforward net
        Yhat = self.feedforward(X)
        #C = cost(Yh, Y)
        
        # Gradient for this test case
        gradient = tuple([np.zeros(w.shape) for w in self.weights])
        
        # Keep chain rule delta
        # Start: dC0/dAL
        delta = self.cost_prime_function(Yhat, Y)
        
        # Loop backwards through layers
        # Neuron current layer = i+1
        for i in range(1,self.L):
            # Append delta
            # dAL/dZL = sigmoid prime
            delta = np.multiply(self.activation_prime_function(self.Zs[-i]), delta)
            
            # Calculate cost gradient
            gradient[-i][True] = np.dot(delta,np.transpose(self.As[-(i+1)])) 
            
            # Append delta
            # dZL/dAL-1 = weights without bias
            delta = np.dot(np.transpose(self.weights[-i][:,:-1]),delta) # (nL-1+1)x1
            
        return (gradient, np.sum(self.cost_function(Yhat, Y)))

    # Mini batch update
    def mini_batch_update(self, batch, learning_rate):
        gradients = []
        costs = []

        # Backpropagate data in batch
        for data,label in batch:
            g,c = self.backpropagate(data, label)
            gradients.append(g)
            costs.append(c)
            
        # Calculate average gradient
        g = tuple([np.mean(x, axis=0) for x in zip(*gradients)])
        c = np.mean(costs)
        
        # Adjust weights and biases
        for W,dW in zip(self.weights,g):
            W[True] = W - learning_rate*dW

        # Return average cost of batch
        return c
            
    
    # Train network
    # Loop over data epohcs times
    # Split data into mini batches
    def train(self, training_data, training_labels, batch_size = 30, epochs = 5, learning_rate = 3.0):

        # Truncate data to match batch size
        training = list(zip(training_data, training_labels))
        if len(training) % batch_size != 0:
            training = training[:-(len(training) % batch_size)]
    
        all_costs = []
        # Repeat over traning data
        for i in range(epochs):
            costs = []
            # Scramble data
            np.random.shuffle(training)

            # Divide trainig data into batches
            batches = [training[i:i+batch_size] for i in range(0, len(training), batch_size)]

            # Loop over batches and update weights
            for batch in batches:
                c = self.mini_batch_update(batch, learning_rate)
                costs.append(c)
                all_costs.append(c)
            print("Epoch " + str(i) + " complete: " + str(np.mean(costs)))
        return all_costs



        
# Calculate sigmoid function for input
# Z: input array
# Return value: array with sigmoid of every value of Z
def sigmoid(Z):
    return 1 / (1 + np.exp(-1*Z))

# Calculaye derivate of sigmoid
# Z: input array
# Return value: array with sigmoid prime of every value of Z
def sigmoid_prime(Z):
    return np.multiply(sigmoid(Z),(1-sigmoid(Z)))

# Calculat cost functions
# Yhat: estimated value
# Y: real value
# Return value: cost of estimation
def cost(Yhat, Y):
    return np.power(Yhat - Y, 2)

# Calculate derivate of cost function
# Yhat: estimated value
# Y: real value
# Return value: cost derivate of estimation
def cost_prime(Yhat, Y):
    return 2*(Yhat - Y)
