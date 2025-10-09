import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor



df=pd.read_csv("D:/ML_course-main/advertising.csv")
print(df.head())
print(df.isnull().sum())

features_df=df.drop('Sales',axis=1)
target_df=df['Sales']
features_df.head()

ob=StandardScaler();
scaled_features=ob.fit_transform(features_df)
#print(scaled_features)

x_scaled=pd.DataFrame(scaled_features)
x_scaled.columns=features_df.columns
print(x_scaled.head(20))

x_train,x_test,y_train,y_test=train_test_split(x_scaled,target_df,test_size=.30,random_state=42)

knn2=KNeighborsRegressor(n_neighbors=3)
knn2.fit(x_train,y_train)

y_trainPred=knn2.predict(x_train)
y_testPred=knn2.predict(x_test)

trainR2=r2_score(y_train,y_trainPred)
print('r2 score of trained: ',trainR2)

testR2=r2_score(y_test,y_testPred)
print('r2 score of tested: ',testR2)
