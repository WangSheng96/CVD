import time
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/new_cardio.csv")
x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)



from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

model = Sequential()
model.add(Dense(16, input_shape=(13,)))
model.add(Dense(64, activation=tf.nn.sigmoid))
model.add(Dense(64, activation=tf.nn.sigmoid))
model.add(Dense(2, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("accuracy:",test_acc)



