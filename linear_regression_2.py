import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

df=pd.read_csv("D:\ML_course-main\Housing.csv")
print(df.head())

df.isnull().sum()

#Data preparation
df.replace(to_replace="yes",value=1,inplace=True)
df.replace(to_replace="no",value=0,inplace=True)
df.replace(to_replace="unfurnished",value=0,inplace=True)
df.replace(to_replace="semi-furnished",value=1,inplace=True)
df.replace(to_replace="furnished",value=2,inplace=True)
print(df.head())


#Train Test Split
x=df.drop('price',axis=1)
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

#Model Training
y_test=np.reshape(y_test,(-1,1))

model=LinearRegression()
model.fit(x_train,y_train)

#print the value of the intercept
intercept=model.intercept_
print('Intercept: ',intercept)

slope=model.coef_
print('Slope: ',slope)

f=x.columns
d=pd.DataFrame({'Features':f,'reg_coefficient':slope})
print(d)

Ypred=model.predict(x_test)
r2=r2_score(y_test,Ypred)
print('r2 score : ',r2)
mse=mean_squared_error(y_test,Ypred)
mae=mean_absolute_error(y_test,Ypred)
print('Mean squared error: ',mse,'\nMean absolute error: ',mae)
