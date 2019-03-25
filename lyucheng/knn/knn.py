from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import json
file_list = [
    "t4",
    "t4-diff",
    "t4-healthy",
    "t4-non-exercises",
    "t4-non-smokers",
    "t4-only-females",
    "t4-only-males",
    "t4-smoking",
    "t4-unhealthy",
]

for file_name in file_list:
    data = np.loadtxt(open(file_name+".csv", "rb"), delimiter=",", skiprows=1)
    x = data[:, :-1]
    y = data[:, -1:].astype(int).ravel()
    f1_scores = {}
    for k in range(1, 101):
        print("processing", k)
        neigh = KNeighborsClassifier(n_neighbors=k)
        kf = KFold(n_splits=5)
        kf.get_n_splits(x)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(x_train, y_train)
            y_pred = neigh.predict(x_test)
            f1 = metrics.f1_score(y_test, y_pred)
            f1_scores[k] = f1
    top_k_scores = sorted(f1_scores.items(), key=lambda x:x[1], reverse=True)[:10]
    file = open("result_"+file_name+".json","w+")
    file.write(json.dumps(top_k_scores))
    file.close()
