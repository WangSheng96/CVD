import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import combinations
from keras import Sequential
from keras.layers import Dense


# COLUMNS = [3,4,5]

res = []
lis = [i for i in range(13)]
for ii in range(1,13):
    for comb in combinations(lis,ii):
        COLUMNS = list(comb)
        data = pd.read_csv("data/new_cardio.csv")
        x,y = data.iloc[:,COLUMNS],data.iloc[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

        model = Sequential()
        model.add(Dense(13, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        res.append((comb,(model.score(x_test,y_test))))
        print(comb)
    print(ii)
    break

# import pickle
# pickle.dump(res,open("temp2.sav","wb"))

# for comb, r in res:
#     print(comb, r)

