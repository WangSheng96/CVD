import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import decomposition, metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

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
    for k in range(100,2000,100):
        print("processing", k)
        neigh = xgb.XGBRegressor(objective ="binary:logistic", n_estimators = k, learning_rate=0.01)
        kf = KFold(n_splits=5)
        kf.get_n_splits(x)
        for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                
                train = np.c_[x_train, y_train]
                one_dp_dataset = np.around(train, decimals=1)
                d = {}
                for i,row in enumerate(one_dp_dataset):
                        x_tuple = tuple(row[:-1])
                        yy = int(row[-1])
                        try:
                                d[x_tuple][yy] += 1
                        except:
                                d[x_tuple] = [0,0]
                                d[x_tuple][yy] += 1
                train1 = []

                magic = lambda lis:1 if lis[1]>=lis[0] else 0
                for key in d:
                        xx = list(key)
                        yy = magic(d[key])
                        train1.append(xx+[yy])
                        
                ndtrain = np.array(train1)
                # ndtrain = pd.DataFrame(train1)
                new_x_train = ndtrain[:,:-1]
                new_y_train = ndtrain[:,-1]



                neigh.fit(new_x_train, new_y_train)

                y_pred = neigh.predict(x_test)
                round = lambda num:1 if num > 0.5 else 0
                rounded_y_pred = []
                for item in y_pred:
                    if item > 0.5 :
                        rounded_y_pred.append(1)
                    else:
                        rounded_y_pred.append(0)
                
                recall = metrics.recall_score(y_test, rounded_y_pred)
                recall_scores[k] = recall 

                f1 = metrics.f1_score(y_test, rounded_y_pred)
                f1_scores[k] = f1

                precision = metrics.precision_score(y_test, rounded_y_pred)
                precision_scores[k] = precision 

                accuracy = metrics.accuracy_score(y_test, rounded_y_pred)
                score_scores[k] = accuracy 

    print (recall_scores)
    top_k_recall = sorted(recall_scores.items(), key=lambda x:x[1], reverse=True)[:20]
    top_k_f1 = sorted(f1_scores.items(), key=lambda x:x[1], reverse=True)[:20]
    top_k_precision = sorted(precision_scores.items(), key=lambda x:x[1], reverse=True)[:20]
    top_k_score = sorted(score_scores.items(), key=lambda x:x[1], reverse=True)[:20]
    file = open("result_"+file_name+".json","w+")
    file.write("topk accuracy")
    file.write(json.dumps(top_k_score))
    file.write("topk recall")
    file.write(json.dumps(top_k_recall))
    file.write("topk f1")
    file.write(json.dumps(top_k_f1))
    file.write("topk precision")
    file.write(json.dumps(top_k_precision))
    # file.write("accuracy")
    # file.write(json.dumps(score_scores))
    # file.write("recall")
    # file.write(json.dumps(recall_scores))
    # file.write("f1")
    # file.write(json.dumps(f1_scores))
    # file.write("precision")
    # file.write(json.dumps(precision_scores))
    file.close()

