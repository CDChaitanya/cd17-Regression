# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.simplefilter(action= "ignore" ,category=FutureWarning)
warnings.filterwarnings(action= "ignore" ,category=RuntimeWarning)

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('State', OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
#Avoiding Dummy Variable Trap
X = X[: , 1:]

'''X_dummy = pd.get_dummies(dataset.State)
X_Final = pd.concat([dataset , X_dummy] , axis= 1)
X_Final = X_Final.drop(['State' , 'Profit' , 'Florida'] , axis= 'columns')
X = X_Final'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2 , random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_test)

#Building the optimal model using Backword Elimination
import statsmodels.api as sm
X= np.append(arr= np.ones((50,1)).astype(int) , values= X , axis=1)

X_opt = X[: , [0,1,2,3,4,5] ]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y , exog = X_opt ).fit()
mdl = regressor_OLS.get_robustcov_results(cov_type='HAC' , maxlags=1)
print(mdl.summary())

X_opt = X[: , [0,1,3,4,5] ]   # 2 IS REMOVED
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y , exog = X_opt ).fit()
mdl = regressor_OLS.get_robustcov_results(cov_type='HAC' , maxlags=1)
print(mdl.summary())

X_opt = X[: , [0,3,4,5] ]   # 1 IS REMOVED
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y , exog = X_opt ).fit()
mdl = regressor_OLS.get_robustcov_results(cov_type='HAC' , maxlags=1)
print(mdl.summary())

X_opt = X[: , [0,4,5] ]   # 3  IS REMOVED
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y , exog = X_opt ).fit()
mdl = regressor_OLS.get_robustcov_results(cov_type='HAC' , maxlags=1)
print(mdl.summary())
