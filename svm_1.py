'''
Program to implement svm using iris dataset and find the accuracy of the algorithm.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/Iris.csv")
features=df.drop('Species',axis=1)
target=df['Species']

xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.2,random_state=42)

scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.fit_transform(xtest)

# build model. You can try different kernels like 'linear', 'rbf', etc.
# for example....model_1 = SVC(kernel='linear')

model_1=SVC(kernel='linear')
model_2=SVC(kernel='rbf')

model_1.fit(xtrain,ytrain)
model_2.fit(xtrain,ytrain)

ypred_1=model_1.predict(xtest)
ypred_2=model_2.predict(xtest)

accuracy_1=accuracy_score(ytest,ypred_1)
accuracy_2=accuracy_score(ytest,ypred_2)
classification_report_1=classification_report(ytest,ypred_1)
classification_report_2=classification_report(ytest,ypred_2)

print("Accuracy for linear kernel:",accuracy_1)
print("Accuracy for rbf kernel:",accuracy_2)
print("classification report for linear kernel:\n",classification_report_1)
print("classification report for rbf kernel:\n",classification_report_2)
