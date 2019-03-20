import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

data = pd.read_csv("lzlclean.csv")
x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# reg = LogisticRegression().fit(x_train, y_train)
# print(reg.score(x_test, y_test))

from keras import Sequential
from keras.layers import Dense

m = Sequential()
m.add(Dense(13,activation="relu"))
m.add(Dense(16, activation="relu"))
m.add(Dense(16, activation="relu"))
m.add(Dense(2, activation="softmax"))
m.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
m.fit(x_train.values, y_train.values, epochs=40)
loss, acc = m.evaluate(x_test.values, y_test.values)

print(loss, acc)
