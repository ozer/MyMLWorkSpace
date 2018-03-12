import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

print(dataset)

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor

DTRegressor = DecisionTreeRegressor(random_state=0)
DTRegressor.fit(X, y)

MyPrediction = DTRegressor.predict(6.5)

# Not usefull for 1D dependent variable data
# its better when you have multiple dimensions

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, DTRegressor.predict(X_grid), color='blue')
plt.title('Decision Tree True or False')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
