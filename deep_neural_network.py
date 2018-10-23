import numpy as np

# Network size
layer_sizes = (784, 16, 16, 10)

# Neurons
# N arrays of size 1 x n_L
neurons = tuple([np.concatenate((np.zeros((y,1),dtype=np.float32),[[1]]),axis=0) for y in layer_sizes])

# Weights
# N-1 matrices of size  n_L-1 x n_L (skipping first layer)
weights = tuple([np.zeros((y,x+1),dtype=np.float32) for x,y in zip(layer_sizes[:-1],layer_sizes[1:])])


# Calculate sigmoid function for input
# Z: input array
# Return value: array with sigmoid of every value of Z
def sigmoid(Z):
    return 1 / (1 + np.exp(-1*Z))


# Forward propagate network
# X: Input data array size of first layer
# Return value: Calculated last layer
def forward_propagate(X):
    global neurons
    global weights
    
    # Set first layer to input
    # Force convert list or array to single column
    # [True] replaces every element but keeps the original object
    neurons[0][:-1] = np.transpose(np.matrix(X).flatten())

    # Go through every layer and calculate neuron activation
    for i in range(len(neurons)-1):
        if i == 0:
            neurons[i+1][:-1] = np.dot(weights[i], np.concatenate((neurons[i][:-1],[[1]]),axis=0))
        else:
            neurons[i+1][:-1] = np.dot(weights[i], np.concatenate((sigmoid(neurons[i][:-1]),[[1]]),axis=0))
    
    return sigmoid(neurons[-1][:-1])


# Calculaye derivate of sigmoid
# Z: input array
# Return value: array with sigmoid prime of every value of Z
def sigmoid_prime(Z):
    return np.multiply(sigmoid(Z),(1-sigmoid(Z)))

# Cost function
def cost(AL, Y):
    return np.power(AL - Y, 2)

def cost_prime(AL, Y):
    return 2*(AL - Y)

# Backpropagate network
# X: training input
# Y: training label
# Return value: gradient
def backpropagate(X,Y):
    global weights
    
    # Force convert list or array to single column
    X = np.transpose(np.matrix(X).flatten())
    Y = np.transpose(np.matrix(Y).flatten())
    
    # Calculate estimation (Y hat)
    Yh = forward_propagate(X)
    C = cost(Yh, Y)
    
    # Errors for this test case
    dCdW = tuple([np.zeros(w.shape) for w in weights])
    
    # Keep chain rule tail
    # Start: dC0/dAL
    dCdZLx = cost_prime(Yh, Y) # nLx1
        
    # Loop backwards through layers
    # Neuron current layer = i+1
    for i in range(1,len(layer_sizes)):
    # Append chain rule
        # dAL/dZL = sigmoid prime
        Z = neurons[-i][:-1] # nLx1
        dCdZLx = np.multiply(sigmoid_prime(Z), dCdZLx) # nLx1
        
        # Calculate w gradient
        dCdWL = dCdW[-i] # nL-1xnL
        # dZL/dWL = AL-1 = sigmoid(ZL-1)
        ZLm1 = neurons[-(i+1)][:-1]

        #Check if last iteration
        if i == len(layer_sizes)-1:
            dZdWL = np.concatenate((ZLm1,[[1]]),axis=0) # (nL-1+1)x1
        else:
            dZdWL = np.concatenate((sigmoid(ZLm1),[[1]]),axis=0) # (nL-1+1)x1
        
        dCdWL[True] = np.matmul(dCdZLx,np.transpose(dZdWL)) # TODO: May have to transpose
        
        
        # Append chain rule
        # dZL/dAL-1 = sigmoid prime
        dZdALm1 = weights[-i][:,:-1] # nLxnL-1 TODO: may have to transpose
        dCdZLx = np.matmul(np.transpose(dZdALm1),dCdZLx) # (nL-1+1)x1
        
    return dCdW


# Train network
# Loop over data epohcs times
# Split data into mini batches
def train(training_data, training_labels, batch_size, epochs, learning_rate):
    global weights

    # Start with random weights and biases
    # Random number matrix of same shape
    for w in weights:
        w[True] = np.random.randn(*w.shape)

    # Divide trainig data into batches
    training = list(zip(training_data, training_labels))
    if len(training) % batch_size != 0:
        training = training[:-(len(training) % batch_size)]
    
    costs = []
    for i in range(epochs):
        epoch_costs = []
        np.random.shuffle(training)
        
        batches = [training[i:i+batch_size] for i in range(0, len(training), batch_size)]
        
        for batch in batches:
            dCdWs = []
            Cs = []
            for data,label in batch:
                dCdWs.append(backpropagate(data,label))
                Cs.append(np.sum(cost(forward_propagate(data),label)))
            
            # Calculate average gradient
            dCdW = tuple([np.mean(x,axis=0) for x in zip(*dCdWs)])
            C = np.mean(Cs)
        
            # Adjust weights and biases
            for W,dW in zip(weights,dCdW):
                W[True] = W - learning_rate*dW
    
            costs.append(C)
            epoch_costs.append(C)
        print("Epoch " + str(i) + " complete: " + str(np.mean(epoch_costs)))
    return costs


if __name__ == "__main__":
    size = 10000
    data = [np.random.rand(layer_sizes[0],1) for i in range(size)]
    labels = [np.random.rand(layer_sizes[-1],1) for i in range(size)]
    batch_size = 99
    learning_rate = 0.1
    
    print("Testing functions:")
    print("Train")
    train(data, labels, batch_size, learning_rate)
    print("Forward propagate")
    print(forward_propagate([np.random.rand() for j in range(layer_sizes[0])]))

    
