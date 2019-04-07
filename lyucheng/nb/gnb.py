from sklearn.naive_bayes import MultinomialNB
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
    file_score_dic = {}
    f1_scores = []
    precision_scores = []
    accuracy_scores = []
    recall_scores= []
    kf = KFold(n_splits=5)
    kf.get_n_splits(x)
    counter = 0
    for train_index, test_index in kf.split(x):
        print("processing:" + str(counter))
        counter += 1
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gnb = MultinomialNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores.append(f1)
        precision = metrics.precision_score(y_test, y_pred)
        precision_scores.append(precision)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        recall = metrics.recall_score(y_test, y_pred)
        recall_scores.append(recall)
    f1_average = sum(f1_scores)/len(f1_scores)
    precision_average = sum(precision_scores)/len(precision_scores)
    accuracy_average = sum(accuracy_scores)/len(accuracy_scores)
    recall_average = sum(recall_scores)/len(recall_scores)
    file_score_dic["f1"] = f1_average
    file_score_dic["precison"] = precision_average
    file_score_dic["accuracy"] = accuracy_average
    file_score_dic["recall"] = recall_average
    file = open("result.json", "w+")
    file.write(json.dumps(file_score_dic))
    file.close()
