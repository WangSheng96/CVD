# classification using regression 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("data/cardio_1dp.csv")
test = pd.read_csv("data/new_cardio.csv")
x_train,y_train = data.iloc[:,:-1], data.iloc[:,-1]
x_test,y_test = test.iloc[:,:-1], test.iloc[:,-1]

results = []
r = lambda x:round(x,4)

from time import time
def cp(t,string="",end="-"*50):
    now = time(); print(end,string,"-->",round(now-t[-1],6),"seconds",end); t.append(now) 

t = [time()]
for title, reg in [     ("linear regression",LinearRegression()),
                        ("ridge alpha = 0.001",Ridge(alpha=0.001)),
                        ("ridge alpha = 0.01",Ridge(alpha=0.01)),
                        ("ridge alpha = 0.1",Ridge(alpha=0.1)),
                        ("ridge alpha = 1",Ridge(alpha=1)),
                        ("ridge alpha = 10",Ridge(alpha=10)),
                        ("ridge alpha = 10",Ridge(alpha=10)),
                        ("ridge alpha = 100",Ridge(alpha=100)),
                        ("lasso alpha = 0.001",Lasso(alpha=0.001)),
                        ("lasso alpha = 0.01",Lasso(alpha=0.01)),
                        ("lasso alpha = 0.1",Lasso(alpha=0.1)),
                        ("lasso alpha = 1",Lasso(alpha=1)),
                        ("lasso alpha = 10",Lasso(alpha=10)),
                        ("lasso alpha = 100",Lasso(alpha=100)),
                        ("elastic net alpha = 0.001 l1_ratio = 0.1", ElasticNet(alpha=0.001, l1_ratio=0.1)),
                        ("elastic net alpha = 0.001 l1_ratio = 0.5", ElasticNet(alpha=0.001, l1_ratio=0.5)),
                        ("elastic net alpha = 0.001 l1_ratio = 0.9", ElasticNet(alpha=0.001, l1_ratio=0.9)),
                        ("elastic net alpha = 0.01 l1_ratio = 0.1", ElasticNet(alpha=0.01, l1_ratio=0.1)),
                        ("elastic net alpha = 0.01 l1_ratio = 0.5", ElasticNet(alpha=0.01, l1_ratio=0.5)),
                        ("elastic net alpha = 0.01 l1_ratio = 0.9", ElasticNet(alpha=0.01, l1_ratio=0.9)),
                        ("elastic net alpha = 0.1 l1_ratio = 0.1", ElasticNet(alpha=0.1, l1_ratio=0.1)),
                        ("elastic net alpha = 0.1 l1_ratio = 0.5", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                        ("elastic net alpha = 0.1 l1_ratio = 0.9", ElasticNet(alpha=0.1, l1_ratio=0.9)),
                        ("elastic net alpha = 1 l1_ratio = 0.1", ElasticNet(alpha=1, l1_ratio=0.1)),
                        ("elastic net alpha = 1 l1_ratio = 0.5", ElasticNet(alpha=1, l1_ratio=0.5)),
                        ("elastic net alpha = 1 l1_ratio = 0.9", ElasticNet(alpha=1, l1_ratio=0.9)),
                        ("elastic net alpha = 10 l1_ratio = 0.1", ElasticNet(alpha=10, l1_ratio=0.1)),
                        ("elastic net alpha = 10 l1_ratio = 0.5", ElasticNet(alpha=10, l1_ratio=0.5)),
                        ("elastic net alpha = 10 l1_ratio = 0.9", ElasticNet(alpha=10, l1_ratio=0.9)),
                        ("elastic net alpha = 100 l1_ratio = 0.1", ElasticNet(alpha=100, l1_ratio=0.1)),
                        ("elastic net alpha = 100 l1_ratio = 0.5", ElasticNet(alpha=100, l1_ratio=0.5)),
                        ("elastic net alpha = 100 l1_ratio = 0.9", ElasticNet(alpha=100, l1_ratio=0.9))
                    ]: 
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    y_test = [0 if i<0.5 else 1 for i in y_test]
    y_pred = [0 if i<0.5 else 1 for i in y_pred]
    scores = {
        "accuracy score:": r(accuracy_score(y_test, y_pred)),
        "precision score:": r(precision_score(y_test,y_pred)),
        "recall score:": r(recall_score(y_test,y_pred)),
        "f1 score:": r(f1_score(y_test,y_pred))
    }
    results.append((title,scores)); cp(t,string="training "+title)

print()
for t,s in results:
    print(t," "*(40-len(t)),s)
print()

