import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

print(dataset)

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape((-1, 1))))

from sklearn.svm import SVR

MySVR = SVR(kernel='rbf')
MySVR.fit(X, y)
MyPrediction = sc_y.inverse_transform(MySVR.predict(sc_X.transform(np.array([[6.5]]))))

sc_y.inverse_transform(MySVR.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X, y, color='red')
plt.plot(X, MySVR.predict(X), color='blue')
plt.title('SVR')
plt.xlabel('Position Level')
plt.ylabel('Amount of Salary')
plt.show()
