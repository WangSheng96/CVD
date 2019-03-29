import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from helper import *

data = pd.read_csv("data/cardio_1dp.csv")
x_train,y_train = data.iloc[:,:-1], data.iloc[:,-1]

testdata = pd.read_csv("new_cardio.csv")
x_test, y_test = testdata.iloc[:,:-1], testdata.iloc[:,-1]

import tensorflow as tf
from keras.layers import Dense
from keras import Sequential


model = Sequential()
model.add(Dense(x_train.shape[1], activation="relu"))
for num in [64,64]:
    model.add(Dense(num, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
model.fit(x_train.values, y_train.values, epochs=20)
y_pred = model.predict(x_test)


#####
FLOOR, CEILING = 0, 1

temp = [0,0]
ypred = y_pred
y_pred = [getClass(i, floor=FLOOR, ceiling=CEILING, record=temp) for i in y_pred]

scores = {"accuracy":round(accuracy_score(y_test, y_pred),4), "precision":round(precision_score(y_test,y_pred),4), "recall":round(recall_score(y_test,y_pred),4), "f1 score":round(f1_score(y_test,y_pred),4)}

print()

print(scores)

print(temp, temp[0]*100/temp[1],"%")

import matplotlib.pyplot as plt

for i in range(1,11):
    th = i/10
    lis = [y for y in ypred if y<th and y>th-0.1]
    print(i, len(lis),"/",len(ypred), "\t\t", len(lis)*100/len(ypred),"%")

