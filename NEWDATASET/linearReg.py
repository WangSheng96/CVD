import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/cardio_1dp.csv")

x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

y_test = [0 if i<0.5 else 1 for i in y_test]
y_pred = [0 if i<0.5 else 1 for i in y_pred]

print(accuracy_score(y_test, y_pred))

