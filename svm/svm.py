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

#SUPPORT VECTOR MACHINE
C = [0.001,0.01,0.1,1,10,100,1000,10000]
C = [10] #C==10 is selected to be the best
svm_score = []
X_train_sample = X_train#[:10000]
y_train_sample = y_train#[:10000]
X_test_sample = X_test
y_test_sample = y_test

for C in C:
    from sklearn.svm import SVC
    regressor = SVC(kernel='rbf', C=C, decision_function_shape='ovr')
    regressor.fit(X_train_sample, y_train_sample)
    y_pred = regressor.predict(X_test_sample)
    result = np.zeros([len(y_pred),2])
    for i in range(len(y_pred)):
        result[i][0] = y_pred[i]
        result[i][1] = y_test_arr[i]
    score = regressor.score(X_test_sample, y_test_sample)
    svm_score.append(score)

print("Accuracy score for the above model at C=10 is:",svm_score)