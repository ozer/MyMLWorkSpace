import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
dataset = pd.read_csv("/Users/ozercevikaslan/Desktop/MachineLearning/Data_Preprocessing/Data.csv")

X = dataset.iloc[:,:-1].values
print("initial X ",X)
Y = dataset.iloc[:,3].values

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
# all X taking all the lines and Y indexes 1,2
imputer = imputer.fit(X[:,1:3])

X[:, 1:3] = imputer.transform(X[:,1:3])

print("X ",X)

labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(Y)

print("Latest X ",X)
print("Latest Y ",y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,y, test_size=0.2,random_state=0
)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("X train : ",X_train)