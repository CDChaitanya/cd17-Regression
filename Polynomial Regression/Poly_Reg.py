# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1 : 2].values  # JUST TO MAKE IT MATRIX NOT VECTOR
y = dataset.iloc[: , 2].values

#AS OUR DATASET IS TOO SMALL WE DONT HAVE TO SPLIT OUR DATA

#FITTING LINEAR REG TO DATASET
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X , y)

#FITTING POLYNOMIAL REG TO DATASET
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)    # fit_transform because first of all we have to fit it to X and then transform it to X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly , y)

#Visualizing Linear Reg Results
plt.scatter(X, y , color='red')
plt.plot(X , lin_reg.predict(X) ,color='blue')
plt.title("TRUTH OR BLUFF (LINEAR MODEL)")
plt.xlabel("POSITION LEVEL")
plt.ylabel("SALARY")
plt.show()

#Visualizing Polynomial Reg Results

#X_grid = np.arange(min(X) , max(X) , step=0.1)
#X_grid = X_grid.reshape( (len(X_grid) , 1) ) #AS THIS WAS A VECTOR WE MUST CONVERT IT TO MATRIX

plt.scatter(X, y , color='red')
plt.plot(X , lin_reg_2.predict(poly_reg.fit_transform(X)) ,color='blue')
plt.title("TRUTH OR BLUFF (POLYNOMIAL MODEL)")
plt.xlabel("POSITION LEVEL")
plt.ylabel("SALARY")
plt.show()

#Predicting a new result with Linear Regression   (X = 6,5)
print( lin_reg.predict(np.array([6.5]).reshape(1, 1)) )
 
#Predicting a new result with Polynomial Regression   (X = 6,5)
print( lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1))) )