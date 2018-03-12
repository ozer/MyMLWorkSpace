import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('kc_house_data.csv')

dataset.drop('zipcode', axis=1, inplace=True)
dataset.drop('lat', axis=1, inplace=True)
dataset.drop('long', axis=1, inplace=True)

X = dataset.iloc[:, 3:18].values

y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestRegressor

RFRegressor = RandomForestRegressor(n_estimators=500, random_state=0)
RFRegressor.fit(X_train, y_train)

haha = X_test[2, :]

result = RFRegressor.predict([haha])

X_grid = np.arange(max(X_test), min(X_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 15))
plt.scatter(X_train[:, 0], y_train, color='red')
plt.plot(X_grid, RFRegressor.predict(X_grid), color='blue')
plt.title('Decision Tree')
plt.xlabel('X Variables')
plt.ylabel('Price')
plt.show()
