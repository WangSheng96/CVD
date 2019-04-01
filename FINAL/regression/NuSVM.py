import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from time import time
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


t = [time()]
def cp(t,string="",end="-"*50):
    now = time(); print(end,string,"-->",round(now-t[-1],6),"seconds",end); t.append(now) 

def r(n, dp=4):
    return round(n,dp)

data = pd.read_csv("../data/cardio_1dp.csv")
test = pd.read_csv("../data/new_cardio.csv")
x_train, y_train = data.iloc[:,:-1], data.iloc[:,-1]
x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

results = []
# trainign starts here
for NU in [0.3, 0.5, 0.7, 0.9]:
    model = NuSVR(nu=NU, gamma="scale")
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    y_pred = [0 if i<0.5 else 1 for i in y_pred]
    scores = {"accuracy":r(accuracy_score(y_test, y_pred)), "precision":r(precision_score(y_test, y_pred)), "recall":r(recall_score(y_test, y_pred)), "f1 score":r(f1_score(y_test, y_pred))} 

    results.append((NU,scores))

for n,s in results:
    print(n," "*27,s)

















