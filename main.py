import network
import matplotlib.pyplot as plt
import struct
import numpy as np

raw_training_data = open('data/train-images-idx3-ubyte','rb')
raw_training_labels = open('data/train-labels-idx1-ubyte','rb')
raw_testing_data = open('data/t10k-images-idx3-ubyte','rb')
raw_testing_labels = open('data/t10k-labels-idx1-ubyte','rb')

# Read data from files into arrays
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

    magic_number = struct.unpack('>I', raw_labels.read(4))[0]
    number_of_items = struct.unpack('>I', raw_labels.read(4))[0]

    n = number_of_items
    labels = struct.unpack(n*'B', raw_labels.read(n))

    to_one_hot = {}
    number_of_labels = len(set(labels))
    sorted_labels = sorted(list(set(labels)))
    for i,l in enumerate(sorted_labels):
        to_one_hot[l] = number_of_labels*[0]
        to_one_hot[l][i] = 1
        to_one_hot[l] = np.array(to_one_hot[l],dtype=np.float32)
        
    labels = [to_one_hot[l] for l in labels]
    
    return (data, labels, tuple(sorted_labels))
    



print('Reading data..')
training_data = preprocess_data(raw_training_data, raw_training_labels)
testing_data = preprocess_data(raw_testing_data, raw_testing_labels)

print('Training...')
net = network.Network((784, 16, 16, 10))
net.train(training_data[0], training_data[1], 10, 10, 3.0)

print('Validating')
lookup = training_data[2]
correct = 0
total = 0
for data, label in zip(training_data[0], training_data[1]):
    oha = net.feedforward(data)
    prediction = lookup[np.argmax(oha)]
    answer = lookup[np.argmax(label)]
    if answer == prediction:
        correct += 1
    total += 1
print("Accuracy training data: " + str(int(100*correct/total)) + "%")

lookup = testing_data[2]
correct = 0
total = 0
for data, label in zip(testing_data[0], testing_data[1]):
    oha = net.feedforward(data)
    prediction = lookup[np.argmax(oha)]
    answer = lookup[np.argmax(label)]
    if answer == prediction:
        correct += 1
    total += 1
print("Accuracy testing data: " + str(int(100*correct/total)) + "%")



