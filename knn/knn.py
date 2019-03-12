from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy
data = numpy.loadtxt(open("./knn/t4-healthy.csv", "rb"), delimiter=",", skiprows=1)
neigh = KNeighborsClassifier(n_neighbors=3)
x = data[:,:-1]
y = data[:,-1:].astype(int).ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
neigh.fit(x_train, y_train)

# Predicted class
# print(neigh.predict(test))


# 3 nearest neighbors
# print(neigh.kneighbors(test)[1])