import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score, precision_score, \
    recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

df=pd.read_csv("C:/Users/mutho/OneDrive/Desktop/DataSet/social-network-ads - social-network-ads.csv")
print(df.head())

print(df.isnull().sum())

x=df.drop(['User ID','Purchased'],axis=1)
y=df['Purchased']
print(x)

print(x.info())

x_dummy=pd.get_dummies(x)
print(x_dummy.head(5))

x_train,x_test,y_train,y_test=train_test_split(x_dummy,y,test_size=.30,random_state=42)

print('Shape of x_train set: ',x_train.shape)
print('Shape of x_test set: ',x_test.shape)
print('Shape of y_train set: ',y_train.shape)
print('Shape of y_test set: ',y_test.shape)

#model creation
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_trainPred=model.predict(x_train)
y_testPred=model.predict(x_test)

#scoreValue=accuracy_score(x_test,y_test)
testScore=model.score(x_test,y_test)
print('test score: ',testScore)

trainScore=model.score(x_train,y_train)
print('train score: ',trainScore)


print(classification_report(y_test,y_testPred))