import pandas as pd
import numpy as np

data = pd.read_csv("cardio_train.csv", sep=";")

#Data Description/Exploration

#age
print("==AGE==")
series = round(data.iloc[:,1]/365)

import collections
counts = collections.Counter(series)
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
plt.show()
print("From the above plot, we can see that the main bulk of the age group ranges from 39 to 65")

#gender
print("==GENDER==")
counts = collections.Counter(data.iloc[:,2])
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
print("Percentage of men and women respectively: ",counts[2]/sum(list(counts.values())), " and ", counts[1]/sum(list(counts.values())));

#height
print("==HEIGHT==")
series = data.iloc[:,3]
mean = np.mean(series)
sd = np.std(series)
counts = collections.Counter(series)
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
plt.show()
print('Height ranges from ',min(keys), 'to', max(keys), 'in cm')

#weight
print("==WEIGHT==")
series = data.iloc[:,4]
mean = np.mean(series)
sd = np.std(series)
counts = collections.Counter(series)
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
plt.show()
print('Weight ranges from ',min(keys), 'to', max(keys), 'in kg')

#systolic blood pressure
print("==SYS BP==")
series = data.iloc[:,5]
counts = collections.Counter(data.iloc[:,5])
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
plt.show()
print('Blood pressure from the original data ranges from ',min(keys), 'to', max(keys), 'in kg')
print('This clearly shows that the data readings have outliers that are extremely errorneous')
print('Data shows that it seems fairly resonable, but upon close observation, it is apparent that values seem to concentrate around 110, 130, 140, 150 and 160.')
print('It also seems pretty clear that readings are supposed to take steps of 10')
print('Hence, we replace all data points that are not in multiples in 10 with mean')
series = data.iloc[:,5]
mean = np.mean([i for i in series if i % 10==0])
series = [mean if i%10!=0 else i for i in series]
counts = collections.Counter(series)
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
plt.show()
print('However, we still witnessed issues with the range. Hence we continue to only accept data between 10 and 240 inclusive')
series = data.iloc[:,5]
accepted = [i for i in series if i%10==0 and i>0 and i<=240]
mean = np.mean([i for i in series if i%10==0 and i>0 and i<=240])
sd = np.std([i for i in series if i%10==0 and i>0 and i<=240])
series = [i if i%10==0 and i>0 and i<=240 else mean for i in series]
counts = collections.Counter(series)
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
series = [i if i%10==0 and i>0 and i<=240 else 0 for i in data.iloc[:,5]]
data.iloc[:,5] = series

#diastolic blood pressure
print("==DIAG BP==")
print("Observing the data, we mean that data points that are not %10==0 and chose the range 10 and 190 inclusive")
series = data.iloc[:,6]
accepted = [i for i in series if i%10==0 and i>0 and i<=190]
mean = np.mean([i for i in series if i%10==0 and i>0 and i<=190])
sd = np.std([i for i in series if i%10==0 and i>0 and i<=190])
series = [i if i%10==0 and i>0 and i<=190 else mean for i in series]
counts = collections.Counter(series)
keys = [i for i in counts.keys()]
values = [i for i in counts.values()]
plt.figure(figsize=(8, 8))
plt.scatter(keys,values)
plt.show()
series = [i if i%10==0 and i>0 and i<=190 else 0 for i in data.iloc[:,6]]
data.iloc[:,6] = series

#cholesterol
print("==CHOLESTEROL==")
series = data.iloc[:,7]
counts = collections.Counter(series)
values = [i for i in counts.values()]
print("1:",counts[1]/sum(list(values)),"2:",counts[2]/sum(list(values)),"3:",counts[3]/sum(list(values)))

#glucose
print("==GLUCOSE==")
series = data.iloc[:,8]
counts = collections.Counter(series)
values = [i for i in counts.values()]
print("1:",counts[1]/sum(list(values)),"2:",counts[2]/sum(list(values)),"3:",counts[3]/sum(list(values)))

#smoke
print("==SMOKE==")
series = data.iloc[:,9]
counts = collections.Counter(series)
values = [i for i in counts.values()]
print("0:",counts[0]/sum(list(values)),"1:",counts[1]/sum(list(values)))

#alcohol
print("==ALCOHOL==")
series = data.iloc[:,10]
counts = collections.Counter(series)
values = [i for i in counts.values()]
print("0:",counts[0]/sum(list(values)),"1:",counts[1]/sum(list(values)))

#active
print("==ACTIVE==")
series = data.iloc[:,11]
counts = collections.Counter(series)
values = [i for i in counts.values()]
print("0:",counts[0]/sum(list(values)),"1:",counts[1]/sum(list(values)))

#target
print("==TARGET==")
series = data.iloc[:,12]
counts = collections.Counter(series)
values = [i for i in counts.values()]
print("0:",counts[0]/sum(list(values)),"1:",counts[1]/sum(list(values)))

#================================================================#
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

#ORDINARY LINEAR REGRESSION
from sklearn import linear_model
import statsmodels.api as sm
regressor = linear_model.LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_test_arr = np.array(y_test)
result = np.zeros([len(y_pred),2])
for i in range(len(y_pred)):
    result[i][0] = y_pred[i]
    result[i][1] = y_test_arr[i]
regressor_OLS = sm.OLS(y_train, X_train).fit()
regressor_OLS.summary()

from sklearn import metrics
mse_linear = metrics.mean_squared_error(y_test, y_pred)
mse = y_pred - y_test
mse = sum(mse**2)
mse = mse/len(y_pred)
print("MSE of the linear regression on test data:", mse_linear)

#LASSO LINEAR REGRESSION
alphas = [0.001,0.01,0.1,1]
lasso_mse = []
lasso_model_complexity = []
lasso_reg_cost = []
for i in alphas:
    regressor = linear_model.Lasso(alpha=i)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    y_train_pred = regressor.predict(X_train)
    mse = metrics.mean_squared_error(y_test, y_pred)
    lasso_mse.append(mse)
    complexity = np.sqrt(sum(regressor.coef_**2))
    lasso_model_complexity.append(complexity)
    lasso_reg_cost.append(mse + i * complexity)
    
#RIDGE LINEAR REGRESSION
alphas = [0.001,0.01,0.1,1]
ridge_mse = []
ridge_model_complexity = []
ridge_reg_cost = []
for i in alphas:
    regressor = linear_model.Ridge(alpha=i)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    ridge_mse.append(mse)
    complexity = np.sqrt(sum(regressor.coef_**2))
    ridge_model_complexity.append(complexity)
    ridge_reg_cost.append(mse + i * complexity)
    
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

num_C = 10
C = [1.0] * num_C
for i in range(num_C):
    C[i] = pow(10, i-3)

inv_log_likelihood_train = [0.0] * num_C
inv_log_likelihood_test = [0.0] * num_C
logistic_regression_accuracy_score = []

for i in range(num_C):
    regressor = LogisticRegression(random_state=0, C=C[i])
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    test_accuracy_score = regressor.score(X_test,y_test)
    inv_log_likelihood_test[i] = test_accuracy_score
    training_accuracy_score = regressor.score(X_train,y_train)
    inv_log_likelihood_train[i] = training_accuracy_score
    logistic_regression_accuracy_score.append(test_accuracy_score)
    
inv_log_likelihood_train = 1/np.log(inv_log_likelihood_train) * -1
inv_log_likelihood_test = 1/np.log(inv_log_likelihood_test) * -1

import matplotlib.pyplot as plt
plt.figure(figsize = (6, 6))
plt.xscale('log')
plt.plot(C, inv_log_likelihood_train, 'bo-', C, inv_log_likelihood_test, 'ro-')
plt.show()

#DESCISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
result = np.zeros([len(y_pred),2])
for i in range(len(y_pred)):
    result[i][0] = y_pred[i]
    result[i][1] = y_test_arr[i]
result = [i for i in result if i[0]==i[1]]
accuracy_score = len(result)/len(y_pred)
print("The accuracy for decision tree is:", accuracy_score)

#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
result = np.zeros([len(y_pred),2])
for i in range(len(y_pred)):
    result[i][0] = y_pred[i]
    result[i][1] = y_test_arr[i]
mse = metrics.mean_squared_error(y_test, y_pred)
print("The mse for random forest is:", mse)

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