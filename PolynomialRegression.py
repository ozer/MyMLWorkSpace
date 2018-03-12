import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''
we are not splitting since we have few amount of data
from sklearn.model_selection import train_test_split

# splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''

#  Linear regression for reference space
from sklearn.linear_model import LinearRegression

LNR = LinearRegression()
LNR.fit(X, y)

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

Polly = PolynomialFeatures(degree=4)
Polly_Reg = Polly.fit_transform(X)
Lin_Reg = LinearRegression()
Lin_Reg.fit(Polly_Reg, y)

# Visualize the Linear Reg
plt.scatter(X, y, color='red')
plt.plot(X, LNR.predict(X), color='blue')
plt.title("Truth or Bluff")
plt.xlabel("Position Level")
plt.ylabel("Income")
plt.show()

# Visualize the Polynomial
plt.scatter(X, y, color='red')
plt.plot(X, Lin_Reg.predict(Polly_Reg), color='blue')
plt.title("Polynomial")
plt.xlabel("Position Level")
plt.ylabel("Income")
plt.show()

Lin_Reg.predict(Polly.fit_transform(6.5))