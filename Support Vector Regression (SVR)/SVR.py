# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1 : 2].values  # JUST TO MAKE IT MATRIX NOT VECTOR
y = dataset.iloc[: , 2].values

#AS OUR DATASET IS TOO SMALL WE DONT HAVE TO SPLIT OUR DATA

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(y).reshape(-1,1)
y = sc_y.fit_transform(y)
y = y.flatten()


#FITTING SVM TO DATASET
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf = Gaussian
regressor.fit(X , y)
 
#Predicting a new result with SVR   (X = 6,5)
#y_pred = regressor.predict( np.array([6.5]).reshape(1, 1) ) 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing Regression Results
plt.scatter(X, y , color='red')
plt.plot(X , regressor.predict(X) ,color='blue')
plt.title("TRUTH OR BLUFF (SVR)")
plt.xlabel("POSITION LEVEL")
plt.ylabel("SALARY")
plt.show()

#Visualizing Regression Results (FOR HIGHER RESOLUTION AND SMOTHER RESULT)
X_grid = np.arange(min(X) , max(X) , step=0.1)
X_grid = X_grid.reshape( (len(X_grid) , 1) ) #AS THIS WAS A VECTOR WE MUST CONVERT IT TO MATRIX

plt.scatter(X, y , color='red')
plt.plot(X_grid , regressor.predict(X_grid) ,color='blue')
plt.title("TRUTH OR BLUFF ( SVR WITH HIGH RESOLUTION)")
plt.xlabel("POSITION LEVEL")
plt.ylabel("SALARY")
plt.show()