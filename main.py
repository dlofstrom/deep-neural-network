from deep_neural_network import *
import matplotlib.pyplot as plt
import struct
import numpy as np

raw_training_data = open('data/train-images-idx3-ubyte','rb')
raw_training_labels = open('data/train-labels-idx1-ubyte','rb')

raw_testing_data = open('data/t10k-images-idx3-ubyte','rb')
raw_testing_labels = open('data/t10k-labels-idx1-ubyte','rb')



def preprocess_data(raw_data, raw_labels):
    # Data
    magic_number = struct.unpack('>I', raw_data.read(4))[0]
    number_of_images = struct.unpack('>I', raw_data.read(4))[0]
    number_of_rows = struct.unpack('>I', raw_data.read(4))[0]
    number_of_columns = struct.unpack('>I', raw_data.read(4))[0]

    data = []
    n = number_of_rows * number_of_columns
    for i in range(number_of_images):
        d = struct.unpack(n*'B', raw_data.read(n))
        d = np.array(d)/256
        data.append(d)
        #print(data)
        

    # Label
    magic_number = struct.unpack('>I', raw_labels.read(4))[0]
    number_of_items = struct.unpack('>I', raw_labels.read(4))[0]

    n = number_of_items
    labels = struct.unpack(n*'B', raw_labels.read(n))

    to_one_hot = {}
    number_of_labels = len(set(labels))
    for i,l in enumerate(sorted(list(set(labels)))):
        to_one_hot[l] = number_of_labels*[0]
        to_one_hot[l][i] = 1
        to_one_hot[l] = np.array(to_one_hot[l])
        
    labels = [to_one_hot[l] for l in labels]
    
    return (data, labels)
    

print('Reading data..')
training_data = preprocess_data(raw_training_data, raw_training_labels)
testing_data = preprocess_data(raw_testing_data, raw_testing_labels)

"""
plt.imshow(training_data[0][0].reshape((28,28)))
plt.show()
print(training_data[1][0])
"""

print('Training...')
C = train(training_data[0], training_data[1], 200, 0.1)

plt.plot(C)
plt.ylabel('Average training cost')
plt.show()


for i in range(10):
    print(forward_propagate(testing_data[0][i]))
    plt.imshow(testing_data[0][i].reshape((28,28)))
    plt.show()    
