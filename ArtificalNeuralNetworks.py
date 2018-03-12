import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  Lets make the Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

#  initializing the Artificial Neural Network
classifier = Sequential()

#  Lets add the input layer and the first hidden layer
# Using Dense for adding Layer
# output_dim parameter in Dense
#  how many nodes ?? average of the input and hidden layer
# the average number of nodes in the input layer 11
# the number of output layer is 1,
# the average of dimension is 11+1 / 2 = 6 !!
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
# adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#  final layer the output layer
# take the care the outcome which is the binary type in this case
#  we are using sigmoid activation function thus the type of the outcome
#  if the outcome's dimensions were more than 1, we would use soft-max instead sigmoid
# and of course, we have to take the average of the outcomes dimensions as units parameter
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# lets go
# optimizer ? algorithm to find the optimal set of weights in the NN
# even though we have different layers, the weights are still only initialized ** ?
#  find the best weights is important which makes it powerful

# loss parameter : loss function( stochastic gradient descent algorithm based on loss function )
# within the stochastic gradient descent to optimize find the optimal weights
# logaritmic loss function
# if the outcome more than 1 dimensional, categorical_crossentropy which is the single in this case

# metrics parameter : criterion to evaluate our model
# typically we use the accuracy criterion to improve models performance
#  the accuracy of the NN will increase little by little each step because we choose the accuracy criterion

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

MyPrediction = classifier.predict(X_test)
# we have to convert the probabilities to true or false by choosing a threshold
MyPrediction = (MyPrediction > 0.5)

from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, MyPrediction)
