from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
data = np.loadtxt(open("./knn/t4-healthy.csv", "rb"), delimiter=",", skiprows=1)
x = data[:, :-1]
y = data[:, -1:].astype(int).ravel()
f1_scores = {}
for k in range(1, 101):
    print("processing", k)
    neigh = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5)
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores[k] = f1
print(f1_scores)
top_k_scores = sorted(f1_scores.items(), key=lambda x:x[1], reverse=True)[:10]
print(top_k_scores)