import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from time import time
from helper import *
from random import random as ran

t = [time()]

data = pd.read_csv("data/cardio_1dp.csv")
data2 = pd.read_csv("new_cardio.csv")

x_train, y_train = data.iloc[:,:-1], data.iloc[:,-1]
x_test, y_test = data2.iloc[:,:-1], data2.iloc[:,-1]

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


cp(t,"importing regressors")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def r(n, dp=4):
    return round(n,dp)

model = LinearRegression()
model.fit(x_train, y_train)

c = list(data.columns)
p = [(i,r(v)) for i,v in enumerate(model.coef_)]

p.sort(key=lambda x:x[1], reverse=True)
print(p)
indexes = [i[0] for i in p]
print(indexes)

output = [c[i] for i in indexes]
print()
for o in output:
    print(o)




















