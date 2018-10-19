from deep_neural_network import *
import matplotlib.pyplot as plt

print("Testing functions:")
print("Train")
C = train()

print("Forward propagate")
X = np.random.rand(layer_sizes[0])
print(forward_propagate(X))


plt.plot(C)
plt.ylabel('Average training cost')
plt.show()
