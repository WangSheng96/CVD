import time
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from os import listdir

i = 0
results = []
for filename in listdir("data"):
    try:
        data = pd.read_csv("data/"+filename)
        x,y = data.iloc[:,:-1], data.iloc[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

        print(x_train.shape)

        model = Sequential()
        model.add(Dense(128, input_shape=(x_train.shape[1],)))
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dense(2, activation=tf.nn.softmax))
        model.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(x_train, y_train, epochs=50)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(filename,"accuracy:",test_acc)

        results.append((filename, test_acc))

    except:
        print("ERROR:        ",filename)

    i += 1

    if i > 2:
        break

print("\n\n\n")
for filename, acc in results:
    print(filename,":",acc)

