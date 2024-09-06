# -*- coding: utf-8 -*-
"""
version: 1.0

@author: Tuo Yang
"""
"""
A handwritten digit recognition app using the MNIST dataset will be implemented
here. A simple CNN will be built to realize the recogition task. In the end, 
a GUI will be built, from which we can draw the digit and recognize it 
straight away
"""

"""
1. Import the libraries and load the dataset
"""
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.losses import categorical_crossentropy

# the data, split between train and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

"""
2. Preprocess the data
The dimension of the training data is (60000, 28, 28). The CNN model will
require one more dimension so we reshape the matrix to shape 
(60000, 28, 28, 1)
"""
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
num_classes = 10

# convert class vectors to binary class metrics(one-hot code)
# e.g., 5 ->[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalization process, making data range restricted within (0, 1)
# as the intensity range of each pixel is (0, 255) 
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

"""
3. Create the model
Creating a self-customized CNN model, which consists of convolutional
and pooling layers, so it works better for data that are represented
as grid structures (e.g, CNN works well for image classification problems)
The dropout layers are used for deactive some neurons during the training,
aiming for preventing overfitting of the model from happening. The whole
model will be complied with Adadelta optimizer
"""
batch_size = 128
epochs = 100

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())      # data are folded into one-dimensional vector
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, 
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

"""
4. Train the model
Using model.fit() function, it takes the training data, validation data,
epochs, and batch size
"""
# verbose: log display
# verbose = 0 no outputs in the console
# verbose = 1 output records of the progress bar
# verbose = 2 output records for each epoch
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 verbose=1, validation_data=(x_test, y_test))
print("The model training is done!")

model.save('C:/Projects to deal with/HandwrittenDigitRecognition/model saved/mnist.h5')
print("The model is saved as mnist.h5")


"""
5. Evaluate the model
Using 10000 images that the training model has never known to
make evaluation 
"""
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss after the last epoch:', score[0])
print('Test accuracy:', score[1])




