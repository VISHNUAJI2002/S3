'''
Problem Statement
A real estate company wishes to analyse the prices of properties based on various
factors such as area, number of rooms, bathrooms, bedrooms, etc. Create a multiple
linear regression model which is capable of predicting the sale price of houses based on
multiple factors and evaluate the accuracy of this model.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

df = pd.read_csv('https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/Housing_Price.csv')
print(df.head())
print(df.isnull().sum())

df.replace(to_replace="yes", value=1, inplace=True)
df.replace(to_replace="no", value=0, inplace=True)
df.replace(to_replace="unfurnished", value=0, inplace=True)
df.replace(to_replace="semi-furnished", value=1, inplace=True)
df.replace(to_replace="furnished", value=2, inplace=True)
print(df.head())

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
random_state=42)

y= np.reshape(y_test,(-1,1))
model = LinearRegression()
model.fit(X_train, y_train)
print('Slope = ',model.coef_)
print('Intercept = ',model.intercept_)
print(' ')
feature = X.columns
coeff = model.coef_
d = pd.DataFrame({'Column':feature, 'Coefficients':coeff})
print(d)

print(' ')
y_pred = model.predict(X_test)
r2 = r2_score(y, y_pred)
mes = mean_squared_error(y, y_pred)
rmes = np.sqrt(mes)
mae = mean_absolute_error(y_pred, y)
print('R2 score = ',r2)
print('Mean Squared Error = ',mes)
print('Root Mean Squared Error = ',rmes)
print('Mean Absolute Error = ',mae)
