#import Modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL.Image import Image
from sklearn import tree    
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score,confusion_matrix


df = pd.read_csv('https://raw.githubusercontent.com/THOMASBAIJU/IRIS_DATASET/refs/heads/main/IRIS.csv')
print(df.head)
print(df.shape)
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

fname= ["sepal_length","sepal_width","petal_length","petal_width"]
tname=y.unique().tolist()

model = DecisionTreeClassifier(criterion="entropy",min_samples_split=15)
model.fit(X_train,y_train)
dot_data = export_graphviz(model,out_file=None,feature_names=fname,class_names=tname,filled=True,rounded=True)
graph = Source(dot_data)
graph.render("Iris_Decision_Tree", view=True,format='png')
print("\n Decision Tree visualization Saved as Iris_Decision_Tree.png")

y_pred = model.predict(X_test)

print("Accuracy : ",accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,5))
sns.heatmap(cm,annot=True)
plt.show()

plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True,fontsize=10)
plt.show()
