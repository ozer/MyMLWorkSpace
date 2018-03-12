from keras.models import load_model
import numpy as np

classifier = load_model('catordog.h5')

categoricalClassifier = load_model('categoricalModel2.h5')

from keras.preprocessing import image as imageHandler

im = imageHandler.load_img('/Users/ozercevikaslan/Desktop/dog2.jpg', target_size=(64, 64))
im = imageHandler.img_to_array(im)
im = np.expand_dims(im, axis=0)
im /= 255.
result = classifier.predict_on_batch(im)

im = imageHandler.load_img('/Users/ozercevikaslan/Desktop/dog4.jpg', target_size=(64, 64))
im = imageHandler.img_to_array(im)
im = np.expand_dims(im, axis=0)
im /= 255.
classes = categoricalClassifier.predict(im)
