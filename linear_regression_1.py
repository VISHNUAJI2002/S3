'''
Problem Statement
The most important factor for an Insurance Company is to determine what premium charges must be paid by an individual. The charges
depend on various factors like age, gender, income, etc.
Build a model that is capable of predicting the insurance charges a person has to pay depending on his/her age using simple l inear
regression. Also, evaluate the accuracy of your model by calculating the value of error metrics such as R-squared, MSE, RMSE, and MAE.

List of Activities:
Activity 1: Analysing the Dataset
Activity 2: Train-Test Split
Activity 3: Model Training
Activity 4: Model Prediction and Evaluation
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Activity 1: Analysing the Dataset

#Print the first five rows of the dataset. Check for null values and treat them accordingly.
df=pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/insurance_dataset.csv")
print(df.head())
print(df.isnull().sum())

#Create a regression plot with age on X-axis and charges on Y-axis to identify the relationship between these two attributes.
plt.figure(figsize=(5,5))
sns.regplot(x="age",y="charges",data=df,color="blue")
plt.title("age-charges plot")
plt.xlabel('age')
plt.ylabel('charges')
plt.grid()
plt.show()

'''
Activity 2: Train-Test Split
We have to determine the effect of age on insurance charges. Thus, age is the feature variable and charges is the target variable.
Split the dataset into training set and test set such that the training set contains 67% of the instances and the remaining instances will
become the test set.
'''

x=df['age']
y=df['charges']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

'''
Activity 3: Model Training
Implement simple linear regression using sklearn module in the following way:
1. Reshape the feature and the target variable arrays into two-dimensional arrays by using reshape(-1, 1) function of numpy module.
2. Deploy the model by importing the LinearRegression class and create an object of this class.
3. Call the fit() function on the LinearRegression object and print the slope and intercept values of the best fit line.
'''

x_train = np.reshape(x_train,(-1,1))
x_test = np.reshape(x_test,(-1,1))
y_train = np.reshape(y_train,(-1,1))
y_test = np.reshape(y_test,(-1,1))

print("x_train shape : ",x_train.shape)
print("x_test shape : ",x_test.shape)
print("y_train shape : ",y_train.shape)
print("y_test shape : ",y_test.shape)

model = LinearRegression()
model.fit(x_train,y_train)
print("Intercept : ",model.intercept_)
print("Co-efficient : ",model.coef_)

'''
Activity 4: Model Prediction and Evaluation
Predict the values for both training and test sets by calling the predict() function on the LinearRegression object. Also, calculate the R2
,

MSE, RMSE and MAE values to evaluate the accuracy of your model.
'''

y_pred = model.predict(x_test)
mn_sq_err = mean_squared_error(y_test,y_pred)
rt_mn_sq_err = np.sqrt(mn_sq_err)
mn_ab_err = mean_absolute_error(y_test,y_pred)
r2_scr = r2_score(y_test,y_pred)
print("Mean squared error : ",mn_sq_err)
print("Root mean squared error : ",rt_mn_sq_err)
print("Mean absolute error : ",mn_ab_err)
print("R2 Score (accuracy) : ",r2_scr)
