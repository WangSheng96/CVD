import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

data = pd.read_csv("../data/new_cardio.csv")
x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

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

y_pred = [0 if i<0.5 else 1 for i in y_pred]

scores = {"accuracy":round(accuracy_score(y_test, y_pred),4), "precision":round(precision_score(y_test,y_pred),4), "recall":round(recall_score(y_test,y_pred),4), "f1 score":round(f1_score(y_test,y_pred),4)}

print(scores)


