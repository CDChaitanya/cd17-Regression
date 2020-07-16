# -*- coding: utf-8 -*-
""" 1. RANDOM FOREST IS A VERSION OF ENSEMBLE LEARNING
    2. ENSEMBLE LEARNING IS WHEN YOU USE MULTIPLE ALGORITHM OR SAME ALGO MULTIPLE TIME
    3. ITS LIKE TEAM OF DT
    4. NON - CONTINOUS MODEL AS THAT OF DT
    5. IF WE SET NUMBER OF TREES TO 10 THEN EACH ONE WILL PREDICT THEIR OWN PREDICTION 
       THEN FOREST TAKES AVG OF THAT THATS WHAT THE SHOW IN THW GRAPH
    6. IF WE INCREASE THE NO OF TREES THAT DOESNT MEANS THAT NUMBER OF STEPS WILL BE INCREASED IN GRAPH
    
### 7.TO INCREASE THE ACCURACY OF THE MODEL ADJUST (INCREASE) NUMBER OF FOREST OR (DECREASE) GRID STEP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1 : 2].values  # JUST TO MAKE IT MATRIX NOT VECTOR
y = dataset.iloc[: , 2].values

#AS OUR DATASET IS TOO SMALL WE DONT HAVE TO SPLIT OUR DATA
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300  ,  criterion ='mse'  , random_state=0)  
regressor.fit(X , y)
 
""" TO INCREASE THE ACCURACY OF THE MODEL ADJUST (INCREASE) NUMBER OF FOREST OR (DECREASE) GRID STEP """

#Predicting a new result with Polynomial Regression   (X = 6,5)
y_pred = regressor.predict( np.array([6.5]).reshape(1, 1) ) 

#Visualizing Regression Results (FOR HIGHER RESOLUTION AND SMOTHER RESULT)
X_grid = np.arange(min(X) , max(X) , step=0.01)
X_grid = X_grid.reshape( (len(X_grid) , 1) ) #AS THIS WAS A VECTOR WE MUST CONVERT IT TO MATRIX

plt.scatter(X, y , color='red')
plt.plot(X_grid , regressor.predict(X_grid) ,color='blue')
plt.title("TRUTH OR BLUFF (RANDOM FOREST WITH HIGH RESOLUTION)")
plt.xlabel("POSITION LEVEL")
plt.ylabel("SALARY")
plt.show()

