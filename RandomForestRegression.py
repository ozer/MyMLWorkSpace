import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

# forest of decision trees

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor

RFRegressor = RandomForestRegressor(n_estimators=300, random_state=0)
RFRegressor.fit(X, y)

MyPrediction = RFRegressor.predict(6.5)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, RFRegressor.predict(X_grid), color='blue')
plt.title('Decision Tree True or False')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
