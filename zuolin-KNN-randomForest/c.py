import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

top = pickle.load(open("temp2.sav","rb"))
top.sort(key=lambda x:x[1], reverse=True)

data = pd.read_csv("data/new_cardio.csv")
y = data.iloc[:,-1]

output = []
activation = "relu"

for a,b in top[:100]:
    COLUMNS = list(a)
    x = data.iloc[:,COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    model = Sequential()
    model.add(Dense(x.shape[1], activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train.values, y_train.values, epochs=20)
    loss, acc = model.evaluate(x_test.values, y_test.values)

    output.append((a,acc))

import pickle

pickle.dump(output,open("NNresults.sav","wb"))
output = pickle.load(open("NNresults.sav","rb"))

for o in output:
    print(o)


