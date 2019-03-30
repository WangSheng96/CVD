import pandas as pd
import numpy as np
data = pd.read_csv("../data/new_cardio.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2019)

#C for regularization
C = [0.001,0.01,0.1,1,10,100,1000,10000]

#SUPPORT VECTOR MACHINE
svm_score = []
svm_recall = []
svm_precision = []
svm_fl_score = []

from sklearn.svm import SVC
from sklearn import metrics

for i in C:
    
    svm_c_score = []
    svm_c_recall = []
    svm_c_precision = []
    svm_c_fl_score = []
    
    for train_index, test_index in kfold.split(X, y):
    
        X_train = X.iloc[train_index][:1000]
        y_train = y.iloc[train_index][:1000]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
    
        classifier = SVC(kernel='rbf', C=i, decision_function_shape='ovr', gamma="scale")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        svm_c_score.append(classifier.score(X_test, y_test))
        svm_c_recall.append(metrics.recall_score(y_test, y_pred))
        svm_c_precision.append(metrics.precision_score(y_test, y_pred))
        svm_c_fl_score.append(metrics.f1_score(y_test, y_pred))
        
    svm_score.append(np.mean(svm_c_score))
    svm_recall.append(np.mean(svm_c_recall))
    svm_precision.append(np.mean(svm_c_precision))
    svm_fl_score.append(np.mean(svm_c_fl_score))

    
print("")
print("=================Results=================")
print("C=\t",C)
print("Score:\t",[str(round(i*100,2)) + "%" for i in svm_score])
print("Recall:\t",[str(round(i*100,2)) + "%" for i in svm_recall])
print("Presc:\t",[str(round(i*100,2)) + "%" for i in svm_precision])
print("F1:\t",[str(round(i*100,2)) + "%" for i in svm_fl_score])
print("")
best_index = np.argmax(svm_score)
print("Best C:",C[np.argmax(svm_score)])
print("Score:",round(svm_score[best_index]*100,2))
print("Recall:",round(svm_recall[best_index]*100,2))
print("Prec:",round(svm_precision[best_index]*100,2))
print("F1:",round(svm_fl_score[best_index]*100,2))

#GAUSSIAN NAIVE BAYES
nb_score = []
nb_recall = []
nb_precision = []
nb_fl_score = []

from sklearn.naive_bayes import GaussianNB

for train_index, test_index in kfold.split(X, y):
    
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    nb_score.append(classifier.score(X_test, y_test))
    nb_recall.append(metrics.recall_score(y_test, y_pred))
    nb_precision.append(metrics.precision_score(y_test, y_pred))
    nb_fl_score.append(metrics.f1_score(y_test, y_pred))

print("")
print("=================Results=================")
print("Score:\t",np.mean(nb_score))
print("Recall:\t",np.mean(nb_recall))
print("Presc:\t",np.mean(nb_precision))
print("F1:\t",np.mean(nb_fl_score))

