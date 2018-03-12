import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Importing packages from keras to build CNN
from keras.models import Sequential
#  Images are 2D
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# First Layer

#  Step #1 - Convolution
# We will add the Convolutional Layer
# Number of features mean the number of feature maps
#  Number of rows and number of columns are represent the feature detector
# border mode is how to feature detectors will handle the vorders of the input image
# most of the time we choose same ( as default )
# input shape**
# it is an important thing since all the images do not same size
#  we work with colored images so 3 channels
#  we will choose 64x64 format because i work with CPU and that still enough to get some good accuracy
#  according to the tensorflow backend, first dimensions, and then channel
#  activation function RELU, we do not want any negative pixel values in our feature maps
#  we also need to remove these negative pixels in order to have non-linearity in our CNN

classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
# Reducing the size of the feature maps
#  Most of the time we take a 2x2 because
#  we do not want to lose the information
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Improvement of the model by going deeper
# what that means is that adding another convolutional layer
classifier.add(Conv2D(32, 3, 3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

#  Step 3 - Flattening
# Taking all our pooled feature maps into single huge vector
# This single vector is going to be the input layer of a feature
#  The high numbers in feature maps got by Max Pooling represents...
# the spatial structure of our images because these high numbers
#  in the feature maps are associated to a specific feature in the input image
classifier.add(Flatten())

# Step 5 - Build basic ANN with fully connected Layers
#  We should not choose too small number for output_dim which is units
# Therefore we need to experiment. Recommended is 128.
#  128 nodes in the hidden layer

classifier.add(Dense(units=128, activation='relu'))

# We need to add output layer
# The outcome is dog or cat, so the dimension of the output is 1
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#  Image augmentation is diverse images by rotating, shifting etc.
# Provides a lot more material to train.
#  We are going to use keras
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
# It is binary thus we have two classes
training_set = train_datagen.flow_from_directory(
    'Convolutional_Neural_Networks/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'Convolutional_Neural_Networks/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

#  Steps per epoch basically means the number of images
#  we have in our training sets
# Since we have 8k images in our training set well,
# here we need to put 8000
# number of epochs
# Validation steps means the number of images in our test set which is 2000
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)

# Options to go deeper in the learning, There are two options
#  First option is to add another convolutional layer
# Second option is to add another fully connected layer
# Adding another convolutional layer makes the most sense

filepath = '/Users/ozercevikaslan/Desktop/catordog.h5'
classifier.save(filepath=filepath)
from keras.preprocessing import image as imageHandler
im = imageHandler.load_img('/Users/ozercevikaslan/Desktop/ben.JPG',target_size=(64,64))
im = imageHandler.img_to_array(im)
im = np.expand_dims(im,axis=0)
im /= 255.
classes = classifier.predict_classes(im, batch_size=32)
print(classes)
