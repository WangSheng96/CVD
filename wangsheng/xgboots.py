import pandas as pd
import numpy as np
from sklearn import decomposition, metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
data = pd.read_csv('https://raw.githubusercontent.com/LyuCheng95/CVD/master/t3-smoking.csv', delimiter = ',')
x,y = data.iloc[:,:-1], data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
for i in range(100,1000,100):
    xg_reg = xgb.XGBRegressor(objective ="binary:logistic", n_estimators = i, learning_rate=0.01)
    xg_reg.fit(x_train,y_train)
    ypred_bst = xg_reg.predict(x_test)
    ypred_bst  = ypred_bst >= 0.5  
    ypred_bst = ypred_bst.astype(int)
    print("i: ",i)
    print("accuracy:", metrics.accuracy_score(y_test,ypred_bst))
    print("recall", metrics.recall_score(y_test,ypred_bst))
    print("f1", metrics.f1_score(y_test,ypred_bst))