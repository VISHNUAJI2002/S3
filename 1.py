import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

df=pd.read_csv("D:/ML_course-main/insurance.csv")
print(df.head())

df.isnull().sum()

plt.figure(figsize=(8,5))
plt.title("Insurance charges and ages")
sns.regplot(x="age",y="charges",color="red",line_kws={'color':'blue'},data=df)
plt.grid()
plt.show()

#splitting into training and testing dataset
x=df['age']
y=df['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)

x_train=np.reshape(x_train,(-1,1))
y_train=np.reshape(y_train,(-1,1))
x_test=np.reshape(x_test,(-1,1))

#creating a model
model=LinearRegression()
model.fit(x_train,y_train)

#displaying slope and intercept

print('Slope: ',model.coef_)
print('Interception', model.intercept_)

#creating prediction variable
pred=model.predict(x_test)

#Evaluating the accuracy
result=r2_score(y_test,pred)
print("Accuracy: ",result)

#Mean squared error
mse=mean_squared_error(y_test,pred)
print("Mean squared error:",mse)

#Rooted mean square error
rmse=root_mean_squared_error(y_test,pred)
print("Rooted mean square error: ",rmse)

#Mean absolute error
mae=mean_absolute_error(y_test,pred)
print("Mean absolute error: ",mae)



