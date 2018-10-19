import numpy as np

# Network size
layer_sizes = (784, 16, 16, 10)

# Neurons
# N arrays of size 1 x n_L
neurons = tuple([np.concatenate((np.zeros((y,1)),[[1]]),axis=0) for y in layer_sizes])

# Weights
# N-1 matrices of size  n_L-1 x n_L (skipping first layer)
weights = tuple([np.zeros((y,x+1)) for x,y in zip(layer_sizes[:-1],layer_sizes[1:])])


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
        neurons[i+1][:-1] = np.matmul(weights[i], sigmoid(neurons[i]))

    return sigmoid(neurons[-1][:-1])


# Calculaye derivate of sigmoid
# Z: input array
# Return value: array with sigmoid prime of every value of Z
def sigmoid_prime(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

# Cost function
def cost(AL, Y):
    return np.power(AL - Y, 2)

def cost_prime(AL, Y):
    return 2*(AL - Y)

# Backpropagate network
# Xs: Batch of training inputs
# Ys: Batch of training labels
# alpha: Learning rate (negative Gradient impact)
# Return value:
def backpropagate(Xs, Ys, alpha):
    global neurons
    global weights
    
    if len(Xs) != len(Ys):
        print('Xs and Ys not same size')
    
    # Error lists filled with individual gradients
    dCdWs = []
    costs = []
    
    while Xs:
        # Pop a test case
        X = Xs.pop(0)
        Y = Ys.pop(0)
        # Force convert list or array to single column
        X = np.transpose(np.matrix(X).flatten())
        Y = np.transpose(np.matrix(Y).flatten())

        # Calculate estimation (Y hat)
        Yh = forward_propagate(X)
        C = cost(Yh, Y)
        
        # Errors for this test case
        # TODO: Automatically the same format
        dCdW = tuple([np.zeros((y,x+1)) for x,y in zip(layer_sizes[:-1],layer_sizes[1:])])
        
        # Keep chain rule tail
        # Start: dC0/dAL
        dCdZLx = cost_prime(Yh, Y) # nLx1
        #print("dCdZLx " + str(dCdZLx.shape))
        
        # Loop backwards through layers
        # Neuron current layer = i+1
        for i in reversed(range(len(layer_sizes)-1)):
            # Append chain rule
            # dAL/dZL = sigmoid prime
            Z = neurons[i+1][:-1] # nLx1
            dCdZLx = np.multiply(sigmoid_prime(Z), dCdZLx) # nLx1
            
            # Calculate w gradient
            dCdWL = dCdW[i] # nL-1xnL
            #print("dCdWL " + str(dCdWL.shape))
            # dZL/dWL = AL-1 = sigmoid(ZL-1)
            ZLm1 = neurons[i][:-1]
            dZdWL = np.concatenate((sigmoid(ZLm1),[[1]]),axis=0) # (nL-1+1)x1
            #print("dZdWL " + str(np.transpose(dZdWL).shape) + " dCdZLxT " + str(dCdZLx.shape))
            #print("dZdWL")
            #print(dZdWL)
            #print("dCdZLx")
            #print(dCdZLx)
            dCdWL[True] = np.transpose(np.matmul(dZdWL,np.transpose(dCdZLx))) # TODO: May have to transpose
            
            
            # Append chain rule
            # dZL/dAL-1 = sigmoid prime
            dZdALm1 = weights[i][:,:-1] # nLxnL-1 TODO: may have to transpose
            dCdZLx = np.matmul(np.transpose(dZdALm1),dCdZLx) # (nL-1+1)x1
            
        # Now append this to dCdWs and dCdBs
        dCdWs.append(dCdW)
        costs.append(np.sum(C))

    # Calculate average gradient
    dCdW = tuple([np.mean(x,axis=0) for x in zip(*dCdWs)])
    C = np.mean(costs)
    
    # Adjust weights and biases
    for W,dW in zip(weights,dCdW):
        W[True] = W - alpha*dW
            
    return C

# Train network
# TODO: implement
def train():
    global neurons
    global weights
        
    X = np.random.rand(layer_sizes[0])
    Y = np.array([1]+[0 for i in range(layer_sizes[-1]-1)])
    n = 10
    Xs = [np.copy(X) for i in range(10)]
    Ys = [np.copy(Y) for i in range(10)]
    costs = []
    
    # Start with random weights and biases
    # Random number matrix of same shape
    weights = tuple([np.random.randn(*w.shape) for w in weights])

    for i in range(1000):
        C = backpropagate(list(Xs), list(Ys), 0.1)
        print("Average cost " + str(C))
        costs.append(C)

    return costs

if __name__ == "__main__":
    ##print("Neurons")
    ##print(neurons)
    ##print("Weights")
    ##print(weights)
    ##print("Biases")
    ##print(biases)

    print("Testing functions:")
    print("Train")
    X = train()
    print("Forward propagate")
    print(forward_propagate(X))
    
