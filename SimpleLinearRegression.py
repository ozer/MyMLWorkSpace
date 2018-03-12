import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns',None)

dataset = pd.read_csv("Salary_Data.csv")

print(dataset.head(100))

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=1/3, random_state=0)


from sklearn.linear_model import LinearRegression

LNR = LinearRegression()
LNR.fit(X_train,Y_train)

Y_prediction = LNR.predict(X_test)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train, LNR.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Amount of Salary')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train, LNR.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Amount of Salary')
plt.show()