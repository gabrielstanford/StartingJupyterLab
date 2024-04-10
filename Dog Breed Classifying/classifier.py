#this code comes entirely from chatgpt when i asked for a dog breed classifier and it literally works. It's not all that accurate but i guess it's a hard problem for a comp to be good at
#next i'll put some effort into understanding it but it all looks pretty familiar. The fiference is in the layers which use all of these different functions which i'm not familiar with yet.

import keras as keras
from keras import layers
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
#this seems to be standard 
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

"x_train has len 50,000 (50,000 inputs), and shape (50,000, 32, 32, 3)"
#I'm pretty sure this refers to 50,000 inputs each with dimensions 32x32px, and each with a red, green, and blue value from 0-255 (this is the last 3 array)
#This is how you use color images - pretty neat!
"y_train has shape (50,000,)"
# Normalize pixel values to between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
#here, we turn all values in each array into floats. This is important for the RGB values, which go from ints from 0-255 into floats from 0-1

# Convert class vectors to binary class matrices
num_classes = 10  # 10 different animal species
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#now our data is preprocessed and ready to enter our net.

# Define the model architecture... now i'm going to turn this from sequential to functional for practice
#filter of conv is the size of the kernel, and the kernel itself is the matrix of values in the filter
model = keras.Sequential([
    keras.Input((32, 32, 3)),
    #reference for creating cnn's. first input is the number of filters, second is the dimensions of the kernel, then the activation
    layers.Conv2D(32, (3, 3), activation='relu'),
    #paramater for maxpooling is just the filter size
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    #this is a typical flow for a convolutional layer which is reusable
    #after a few hidden layers, we flatten into a dense layer 
    #then we have a dense layer (sort of acting as a bridge) before the final output layer which is another normal dense layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()



