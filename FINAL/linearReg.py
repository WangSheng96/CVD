import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("data/cardio_1dp.csv")

x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

for title, reg in [("linear regression",LinearRegression()),("ridge",Ridge()),("lasso",Lasso()),("elastic net", ElasticNet())]: 
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    y_test = [0 if i<0.5 else 1 for i in y_test]
    y_pred = [0 if i<0.5 else 1 for i in y_pred]

    print("-"*50)
    print(title)
    print("accuracy score:",accuracy_score(y_test, y_pred))
    print("precision score:", precision_score(y_test,y_pred))
    print("recall score:", recall_score(y_test,y_pred))
    print("f1 score:", f1_score(y_test,y_pred))
    print("-"*50,"\n")

