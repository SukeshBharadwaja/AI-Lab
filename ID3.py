import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import datasets


iris = datasets.load_iris()
X= iris.data[:,:2]
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

id3_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_tree.fit(X_train,y_train)

pp.figure(figsize=(12,8))
plot_tree(id3_tree, feature_names=iris.feature_names[:2],class_names=iris.target_names, filled=True)
pp.title ("ID3")
pp.show()

accuracy = id3_tree.score(X_test,y_test)
print(f"Accuracy = {accuracy*100:2f}%")


