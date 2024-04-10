#this is essentially where the work will come in. I've got my data loaded in, I understand it, and now I need to translate that into keras.
import keras as keras
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.models import Model
from keras import Input
#We begin with data preprocessing
#it is pretty amazing how much more efficient this is than coding the model ourself. I think now I understand the basics of how to create models, compile them, fit them.
#But I need a lot of work on all the different types of things, but honestly that's the easy part. Tomorrow I'll get going on some new projects I find interesting.

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

#I was trying to do the same thing as earlier, loading in the dataset in that complex way but it turns out that keras makes it incredibly easy to load in mnist
#and other datasets as well, some great things to practice wtih!
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
#x_train has shape (60000, 28, 28) and y_train has shape (10000,)
#below represents a nice and handy way of reshaping the data
num_train  = X_train.shape[0]
num_test   = X_test.shape[0]

img_height = X_train.shape[1]
img_width  = X_train.shape[2]
#this reshape function is a handy way of reshaping data. So instead of 60,000, 28, 28, we'll have 60,000, 784 which is easier to handle
X_train = X_train.reshape((num_train, img_width * img_height))
X_test  = X_test.reshape((num_test, img_width * img_height))
#the to_cateogorical function is a handy way to turn the ints present in y_train into an array of size num_classes (in this case 10), where everything will be 0 except for one place, which will represent the number guess. 
#this is like vectorizing it, so we will be able to get the result with argmax
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

#Now we create our model

"""
#Sequential version for reference
model = Sequential([
    #shape - 60000, img_height*img_width
    Dense(30, activation="sigmoid", name="layer1"),
    Dense(10, activation="sigmoid", name="output")
])
"""

num_classes = 10
input   = Input(shape=(img_height*img_width,))
xo      = Dense(num_classes)(input)
output      = Activation('softmax')(xo)
model   = Model(inputs=[input], outputs=[output])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

#X is input, y is correct output
#X_train has shape (60,000, 784), but we're telling the network it'll have an input shape of (784,). So I suppose model.fit spits into the network one-by-one
"""A tf.data.Dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights)... 
x can be: A NumPy array (or array-like), or a list of arrays (in case the model has multiple inputs)"""
model.fit(X_train, y_train, batch_size=128, epochs=30, verbose=1)

score = model.evaluate(X_test, y_test, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy:', score[1])