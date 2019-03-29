from sklearn.naive_bayes import GaussianNB
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
    "new_cardio"
]

for file_name in file_list:
    data = np.loadtxt(open("./lyucheng-knn-nb/nb/"+file_name+".csv", "rb"), delimiter=",", skiprows=1)
    x = data[:, :-1]
    y = data[:, -1:].astype(int).ravel()
    file_score_dic = {}
    f1_scores = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(x)
    counter = 0
    for train_index, test_index in kf.split(x):
        print("processing:" + str(counter))
        counter += 1
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores.append(f1)
    f1_average = sum(f1_scores)/len(f1_scores)
    file_score_dic[file_name] = f1_average
    file = open("./lyucheng-knn-nb/nb/result.json", "w+")
    file.write(json.dumps(file_score_dic))
    file.close()
