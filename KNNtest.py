import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors=['#FF0000', "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target
print(type(iris.target))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from KNN import KNN
clf = KNN(k=6)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)

#acc = np.sum(predictions == y_test) / len(y_test)
print(clf.accuracy(predictions, y_test))

#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
#plt.show()  