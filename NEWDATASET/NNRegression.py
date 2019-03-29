import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

data = pd.read_csv("data/cardio_1dp.csv")

x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

import tensorflow as tf
from keras.layers import Dense
from keras import Sequential

NN = []
with open("NN.csv") as f:
    for line in f:
        NN.append([int(i) for i in line.strip().split(",") if len(i)>0])

epochs = [5,10,20,30,40,50,100,200,500]

RESULTS = []

for CONFIG in NN:
    for epoch in epochs:
        print("training epochs:",CONFIG,epoch)

        model = Sequential()
        model.add(Dense(x_train.shape[1], activation="relu"))
        for num in CONFIG:
            model.add(Dense(num, activation="relu"))
        model.add(Dense(1, activation="linear"))

        model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
        model.fit(x_train.values, y_train.values, epochs=epoch)
        y_pred = model.predict(x_test)

        y_test = [0 if i<0.5 else 1 for i in y_test]
        y_pred = [0 if i<0.5 else 1 for i in y_pred]

        scores = [accuracy_score(y_test, y_pred), precision_score(y_test,y_pred), recall_score(y_test,y_pred), f1_score(y_test,y_pred)]

        with open("results.csv","a") as f:
            string = f"hidden layers: {CONFIG}, epochs: {epoch} --> \t\taccuracy: {round(scores[0],4)}, precision: {round(scores[1],4)}, recall: {round(scores[2],4)}, f1_score: {round(scores[3],4)}"
            f.write(string + "\n")



