from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

def randomForest(filename, n_estimators=10, criterion="gini", max_depth=None):
    data = pd.read_csv(filename)

    x,y = data.iloc[:,:-1], data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    ascore = accuracy_score(y_test, y_pred)
    pscore = precision_score(y_test, y_pred)
    rscore = recall_score(y_test, y_pred)
    fscore = f1_score(y_test, y_pred)

    SF = 4
    return f"RF  --> accuracy: {round(ascore,SF)}, precision: {round(pscore,SF)}, recall: {round(rscore,SF)}, f1 score: {round(fscore, SF)}", [ascore, pscore, rscore, fscore]

