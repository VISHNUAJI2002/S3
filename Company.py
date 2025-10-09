#implement multiple linear regression using company data set

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error

df=pd.read_csv('D:/ML_course-main/advertising.csv')
print(df.head(10))
df.isnull().sum()

x=df.drop('Sales',axis=1)
y=df['Sales']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

Ypred=model.predict(x_test)

r2=r2_score(Ypred,y_test)
mae=mean_absolute_error(Ypred,y_test)
rms=root_mean_squared_error(Ypred,y_test)

print('r2 score: ',r2*100,'%','\nMean absolute error: ',mae)
print('Mean squared error: ',rms)

c=model.coef_
f=x.columns
d=pd.DataFrame({'Columns':f,"Coef":c})
print(d)

