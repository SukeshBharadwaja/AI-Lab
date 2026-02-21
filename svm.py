import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

X,y = datasets.make_classification(n_samples=100,n_features=2,n_clusters_per_class=1,n_redundant=0,random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
smv=SVC(kernel='linear',C=1.0)
smv.fit(X_train,y_train)

plt.figure(figsize=(8,6))
plot_decision_regions(X,y,clf=smv,legend=2)
plt.xlabel = ("Freature 1")
plt.ylabel = ("Feature 2")
plt.title = ("SVM")
plt.show()

accuracy=smv.score(X_test,y_test)
print(f"modle accuracy: {accuracy*100:.2f}%")