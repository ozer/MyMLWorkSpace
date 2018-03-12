import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")

print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

labelEncoderX = LabelEncoder()
X[:, 3] = labelEncoderX.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.model_selection import train_test_split

# splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear..
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test to compare with true values
y_prediction = regressor.predict(X_test)

from statsmodels.formula import api

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# Backward elimination

'''
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = api.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
'''

'''
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = api.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
'''

'''
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = api.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
'''

'''
X_opt = X[:, [0, 3, 5]]
regressor_OLS = api.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
'''

X_opt = X[:, [0, 3]]
regressor_OLS = api.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

