import numpy as np
# Importing packages from keras to build CNN
from keras.models import Sequential
#  Images are 2D
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, 3, 3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=2, activation='softmax'))

# Metrics will be categorical_accuracy
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'Convolutional_Neural_Networks/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'Convolutional_Neural_Networks/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)

filepath = '/Users/ozercevikaslan/Desktop/categoricalModel2.h5'
classifier.save(filepath=filepath)

from keras.preprocessing import image as imageHandler

im = imageHandler.load_img('/Users/ozercevikaslan/Desktop/dogg.jpg', target_size=(64, 64))
im = imageHandler.img_to_array(im)
im = np.expand_dims(im, axis=0)
im /= 255.
classifier.predict(im)


