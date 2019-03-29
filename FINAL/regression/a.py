import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from time import time
from helper import *
from random import random as ran

t = [time()]

data = pd.read_csv("data/cardio_1dp.csv")
data2 = pd.read_csv("new_cardio.csv")

x_train, y_train = data.iloc[:,:-1], data.iloc[:,-1]
x_test, y_test = data2.iloc[:,:-1], data2.iloc[:,-1]


#[20, 58, 195, 191, 220, 262]:

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=20); cp(t,"preparing data")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

cp(t,"importing regressors")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def r(n, dp=4):
    return round(n,dp)

models = {  "linear regression": LinearRegression(), 
            # "ridge regression": Ridge(alpha=0.1), 
            # "lasso": Lasso(), 
            # "elastic net": ElasticNet(), 
            # "huber regressor (robust to outliers)": HuberRegressor(epsilon=4),
            # "Passive Aggressive Regressor ": PassiveAggressiveRegressor(max_iter=100000, tol=0.5), 
            # "random forest regressor": RandomForestRegressor(n_estimators=10), 
            # "gradient boosting regressor": GradientBoostingRegressor(min_samples_leaf=3),
            # "k nearest neighbiours regressor": KNeighborsRegressor(),
            # "RANSAC regressor": RANSACRegressor(),
            "SGD regressor": SGDRegressor(max_iter=100000, tol=0.5),
            # "kernel ridge": KernelRidge(),
            # "ada boost regressor": AdaBoostRegressor(),
            # "bagging regressor": BaggingRegressor(),
            # "extra trees regressor": ExtraTreesRegressor(n_estimators=10),
            # "dummy regressor": DummyRegressor(),
            # "PLSR regressor": PLSRegression(),
            # "radius neighbours regressor": RadiusNeighborsRegressor(radius=5),
            # "neural_network.MLPRegressor 500": MLPRegressor(hidden_layer_sizes=(50)),
            # "svm.SVR": SVR(gamma="scale"),
            "svm.NuSVR epsilon=": NuSVR(nu=0.7, gamma="scale")
            # "svm.LinearSVR epsilom=": LinearSVR(max_iter=10000)
            # "decision tree regressor": DecisionTreeRegressor(),
            # "extra tree regressor": ExtraTreeRegressor()
        }

# models = {
#             "1":MLPRegressor(hidden_layer_sizes=(64,2), solver="adam"),
#             "2":MLPRegressor(hidden_layer_sizes=(64,2), solver="lbfgs"),
#         }

cp(t, "initialising models")

results = []

rand = [0,0]

for name in models:
    try:
        model = models[name]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        y_pred = [getClass(i, floor=0.45, ceiling=0.55, record=rand) for i in y_pred]

        ascore, pscore, rscore, fscore = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
        # print(name, " "*(60-len(name)),string, " "*(80-len(string)), end=" ") ; 
        results.append((name, ascore, pscore, rscore, fscore))
        cp(t,"training "+name, end="-"*5)
            
    except Exception as e:
        print(name,"ERROR", str(e))

for name, ascore, pscore, rscore, fscore in results:
    string = f"accuracy: {r(ascore)}, precision: {r(pscore)}, recall: {r(rscore)}, f1: {r(fscore)}" 
    print(name, " "*(60-len(name)),string, " "*(80-len(string))) ; 
        
print()
end(t)

print(rand, rand[0]*100/rand[1],"%")


















