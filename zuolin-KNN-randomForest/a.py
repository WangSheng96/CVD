import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv("data/new_cardio.csv")
y = data.iloc[:,-1]

def Model():
    return LogisticRegression(C=0.1)

results = []
for i in range(data.shape[1]-1):
    x = data.iloc[:,i:i+1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model = Model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results.append((i,accuracy_score(y_test, y_pred)))


print("\n"*10)
results.sort(key=lambda x:x[1], reverse=True)
for i,r in results:
    print(i, data.columns[i], r)

