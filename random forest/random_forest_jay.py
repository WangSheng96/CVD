import pandas as pd
import numpy as np
data = pd.read_csv("cardio_train.csv", sep=";")

#problematic blood pressure, lose removal
series = [i if i%10==0 and i>0 and i<=240 else 0 for i in data.iloc[:,5]]
data.iloc[:,5] = series
series = [i if i%10==0 and i>0 and i<=190 else 0 for i in data.iloc[:,6]]
data.iloc[:,6] = series

#removing erroneous entries
data = data.drop(data[data.ap_hi==0].index)
data = data.drop(data[data.ap_lo==0].index)
data = data.drop(data[data.ap_hi<data.ap_lo].index)

#DATA-PREPROCESSING

#GENDER CHANGE 2 TO 1
data.iloc[:,2] = [0 if i==2 else i for i in data.iloc[:,2]]

#ONE HOT ENCODING
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(
    (['age', 'height', 'weight', 'ap_hi', 'ap_lo'], MinMaxScaler()),
    (['cholesterol', 'gluc'], OneHotEncoder())
)
data_transformed = pd.DataFrame(transformer.fit_transform(data))
data_transformed = data_transformed.drop(columns=[7,10]).reset_index()
data_cat = data.iloc[:,[2,9,10,11,12]].reset_index()
data_new = pd.concat([data_transformed,data_cat], axis=1, ignore_index=True)
data_new = data_new.drop(columns=[0,10])
data_new.columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol_0', 'cholesterol_1', 'gluc_0', 'gluc_1', 'gender', 'smoke', 'alco', 'active', 'cardio']

data = data_new

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size = 0.1, random_state = 2009)
y_test_arr = np.array(y_test)

#RANDOM FOREST REGRESSION
C = [10,50,100]
random_forest_mse = []
random_forest_score = []
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
for C in C:   
    regressor = RandomForestRegressor(n_estimators = C, random_state = 0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred_adj = [1 if i>0.5 else 0 for i in y_pred]
    result = np.zeros([len(y_pred),2])
    for i in range(len(y_pred)):
        result[i][0] = y_pred_adj[i]
        result[i][1] = y_test_arr[i]
    mse = metrics.mean_squared_error(y_test, y_pred)
    score = metrics.mean_squared_error(y_test, y_pred_adj)
    random_forest_mse.append(mse)
    random_forest_score.append(score)
    print("The mse for random forest is:", mse)
    print("The accuracy score for random forest after forcing zeros and ones:", score)