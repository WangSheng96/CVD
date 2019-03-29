import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import pickle

data = pd.read_csv("../data/cardio_1dp.csv")
test = pd.read_csv("../data/new_cardio.csv")
x_train,y_train = data.iloc[:,:-1], data.iloc[:,-1]
x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

import tensorflow as tf
from keras.layers import Dense
from keras import Sequential


model = Sequential()
model.add(Dense(x_train.shape[1], activation="relu"))
for num in [64,64]:
    model.add(Dense(num, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
model.fit(x_train.values, y_train.values, epochs=500)
y_pred = model.predict(x_test)

r = lambda x:round(x,4)

evs = explained_variance_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nexplained variance score: {r(evs)}, mean squared error: {r(mse)}, median absolute error: {r(medae)}, mean absolute error: {r(mae)}, r2 score: {r(r2)}\n")






