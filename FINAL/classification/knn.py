from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import json
file_list = [
        "new_cardio"
]

for file_name in file_list:
    data = np.loadtxt(open(file_name+".csv", "rb"), delimiter=",", skiprows=1)
    x = data[:, :-1]
    y = data[:, -1:].astype(int).ravel()
    recall_scores = {}
    f1_scores = {}
    precision_scores = {}
    score_scores = {}
    for k in range(1, 51):
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

            recall = metrics.recall_score(y_test, y_pred)
            recall_scores[k] = recall 

            f1 = metrics.f1_score(y_test, y_pred)
            f1_scores[k] = f1

            precision = metrics.precision_score(y_test, y_pred)
            precision_scores[k] = precision 

            score = neigh.score(x_test, y_test)
            score_scores[k] = precision 

    top_k_recall = sorted(recall_scores.items(), key=lambda x:x[1], reverse=True)[:10]
    top_k_f1 = sorted(f1_scores.items(), key=lambda x:x[1], reverse=True)[:10]
    top_k_precision = sorted(precision_scores.items(), key=lambda x:x[1], reverse=True)[:10]
    top_k_score = sorted(score_scores.items(), key=lambda x:x[1], reverse=True)[:10]
    file = open("result_"+file_name+".json","w+")
    file.write("topk score")
    file.write(json.dumps(top_k_score))
    file.write("topk recall")
    file.write(json.dumps(top_k_recall))
    file.write("topk f1")
    file.write(json.dumps(top_k_f1))
    file.write("topk precision")
    file.write(json.dumps(top_k_precision))
    file.close()
