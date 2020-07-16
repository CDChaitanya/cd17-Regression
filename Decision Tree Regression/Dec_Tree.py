# -*- coding: utf-8 -*-
""" HERE WE HAVE USED DT FOR 1D 
    ITS NOT THAT GOOD FOR 1D
    BUT IT IS BETTER FOR MORE DIMENTIONAL 
    
    AND THIS IS REGRESSION TREES NOT CLASSIFICATION TREE
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1 : 2].values  # JUST TO MAKE IT MATRIX NOT VECTOR
y = dataset.iloc[: , 2].values

#AS OUR DATASET IS TOO SMALL WE DONT HAVE TO SPLIT OUR DATA

#FITTING DECISION TREE TO DATASET
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion="mse" , random_state=0)
regressor.fit(X , y)
 
#Predicting a new result with Polynomial Regression   (X = 6,5)
y_pred = regressor.predict( np.array([6.5]).reshape(1, 1) ) 

"""
plt.scatter(X, y , color='red')
plt.plot(X , regressor.predict(X) ,color='blue')
plt.title("TRUTH OR BLUFF (REGRESSION MODEL)")
plt.xlabel("POSITION LEVEL")                    #ITS JUST PREDICTING AFTER A INTERVAL OF 1
plt.ylabel("SALARY")                            # AS ITS NOT PREDICTING IN BTWEEN VALUES PROPERLY
plt.show()                                      # THIS IS NOT A PROPER DEC TREE
"""
#SO NOW WE HAVE TO INCREASE THE RESOLUTION 0.1 IS NOT ENOUGH SO USE 0.01
X_grid = np.arange(min(X) , max(X) , step=0.01)
X_grid = X_grid.reshape( (len(X_grid) , 1) ) #AS THIS WAS A VECTOR WE MUST CONVERT IT TO MATRIX

plt.scatter(X, y , color='red')
plt.plot(X_grid , regressor.predict(X_grid) ,color='blue')
plt.title("TRUTH OR BLUFF (REGRESSION MODEL WITH HIGH RESOLUTION)")
plt.xlabel("POSITION LEVEL")
plt.ylabel("SALARY")
plt.show()