'''
Aim: Program to implement NaTve Bayes Algorithm using any standard dataset available in the public domain and find the
accuracy of the algorithm.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

df = pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/Iris.csv")
print(df.head())
print(df.isnull().sum())

features = df.drop(['Species'], axis=1)
target = df['Species']
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2,random_state=42,stratify=target)

model = GaussianNB()
model.fit(xtrain, ytrain)

ytrain_pred = model.predict(xtrain)
ytest_pred = model.predict(xtest)

print("Accuracy score of ytest_pred:", accuracy_score(ytest, ytest_pred))
print("\nClassification Report:")
print("Test report:\n", classification_report(ytest, ytest_pred))
print("Confusion Matrix for Test Data:")
print(confusion_matrix(ytest, ytest_pred))
print("Confusion Matrix for Train Data:")
print(confusion_matrix(ytrain, ytrain_pred))

