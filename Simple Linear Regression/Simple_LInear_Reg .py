#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:39:02 2020

@author: chat
"""
#Simple Data Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=1/3 , random_state=0)

#Fitting Simple Linear Regression on Training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualizing the Training set result
plt.scatter(X_train , y_train , color="red")
plt.plot(X_train , regressor.predict(X_train) , color="blue")
plt.title("Salary v/s Years Of Experience (TRAINING)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the Test set result
plt.scatter(X_test , y_test , color="red")
plt.plot(X_train , regressor.predict(X_train) , color="blue")
plt.title("Salary v/s Years Of Experience (TEST)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()