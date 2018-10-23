
# Deep neural network

This is an attempt at designing my own deep neural network

## Theory
The theory in this code comes from the YouTube channel [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw/about)

- [But what *is* a Neural Network? | Deep learning, chapter 1](https://youtu.be/aircAruvnKk)
- [Gradient descent, how neural networks learn | Deep learning, chapter 2](https://youtu.be/IHZwWFHWa-w)
- [What is backpropagation really doing? | Deep learning, chapter 3](https://youtu.be/Ilg3gGewQ5U)
- [Backpropagation calculus | Deep learning, chapter 4](https://youtu.be/tIeHLnjs5U8)

And [this](http://neuralnetworksanddeeplearning.com/chap2.html) article by [Michael Nielsen](http://michaelnielsen.org/)

## Data
Data of handwritten digits are from [The MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)

### Labels

|offset |type           |value            |description              | 
|---|---|---|---|
|0000   |32 bit integer |0x00000801(2049) |magic number (MSB first) |
|0004   |32 bit integer |60000            |number of items          |
|0008   |unsigned byte  |??               |label                    |
|0009   |unsigned byte  |??               |label                    |
|...    |...            |                 |                         |
|xxxx   |unsigned byte  |??               |label                    |

The labels values are 0 to 9.

### Images
|offset |type           |value            |description       | 
|---|---|---|---|
|0000   |32 bit integer |0x00000803(2051) |magic number      |
|0004   |32 bit integer |60000            |number of images  |
|0008   |32 bit integer |28               |number of rows    |
|0012   |32 bit integer |28               |number of columns |
|0016   |unsigned byte  |??               |pixel             |
|0017   |unsigned byte  |??               |pixel             |
|...    |...            |                 |                  |
|xxxx   |unsigned byte  |??               |pixel             |

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).