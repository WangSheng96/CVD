# uses original dataset: new_cardio.csv
# classification using regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

data = pd.read_csv("../data/new_cardio.csv")
x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

import tensorflow as tf
from keras.layers import Dense, Activation
from keras import Sequential

EPOCHS = 2
CONFIG = [64,64]
METRICS = ["sparse_categorical_crossentropy"] #this is the only one that doesnt return an error shag
results = []

def toClass(lis):
    a,b = lis
    return 0 if a>b else 1

for EPOCHS in [50]:
    model = Sequential()
    model.add(Dense(32, activation="relu", input_dim=x_train.shape[1]))
    # for num in CONFIG:
    #     model.add(Dense(num, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train.values, y_train.values, epochs=EPOCHS)
    y_pred = model.predict(x_test)
    y_pred = [toClass(i) for i in y_pred]

    scores = {"accuracy":round(accuracy_score(y_test, y_pred),4), "precision":round(precision_score(y_test,y_pred),4), "recall":round(recall_score(y_test,y_pred),4), "f1 score":round(f1_score(y_test,y_pred),4)}
    results.append((EPOCHS,scores))

for m,s in results:
    print(str(m)," "*29,s)

    