import tensorflow as tf
from keras import Input
from keras.layers import Dense

inputs = Input((4,))
layer1 = Dense(3, activation="relu", name="layer1")
#essentially now we take as input to layer1 the tensor (array) containing 4 neurons. This initializes the weights automatically
x = layer1(inputs)
layer2 = Dense(2, activation="relu", name="layer2")
#^this initializes the layer, giving it weights/biases
#the weights are a tensor with shape (8, 10). This is because each of the 8 input neurons contains a set of 10 weights, corresponding to each neuron in the first layer.
#now we initialize the second layer by taking in the 3 inputs from layer1
y = layer2(x)

#a layer, with inputs passed in, is a tensor consisting of (here) 10 values.
print(layer2.get_weights())



"""layer2 = Dense(3, activation="relu", name="layer2")
layer3 = Dense(4, name="layer3")

print(layer1)
# Call layers on a test input
x = tf.ones((3, 3))
print(x)
print(layer1(x))
y = layer3(layer2(layer1(x)))
"""
