import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from random import random


data = pd.read_csv("data/new_cardio.csv")

smokers = data.query("smoke == 1")
non_smokers = data.query("smoke == 0")

NUM_SMOKERS = smokers.shape[0]
NUM_NON_SMOKERS = non_smokers.shape[0]

select = [True if random()<(NUM_SMOKERS/NUM_NON_SMOKERS) else False for i in range(NUM_NON_SMOKERS)]

toAdd = non_smokers.iloc[select,:]
data = pd.concat([smokers, toAdd])
data = data.sample(frac=1)

x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# reg = LogisticRegression().fit(x_train, y_train)
# print("accuracy:",reg.score(x_test,y_test))

from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(13,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train.values, y_train.values, epochs=20)

loss, acc = model.evaluate(x_test.values, y_test.values)
print(loss, acc)

