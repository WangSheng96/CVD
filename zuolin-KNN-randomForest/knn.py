from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd


def knn(filename, num_neighbours=5, algorithm="auto", leaf_size=30, p=2):
    data = pd.read_csv(filename)

    columns = "COLUMNS:" + " ".join(list(data.columns))
    x,y = data.iloc[:,:-1], data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

    clf = KNeighborsClassifier(n_neighbors=num_neighbours, algorithm=algorithm, leaf_size=leaf_size, p=p)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    ascore = accuracy_score(y_test, y_pred)
    pscore = precision_score(y_test, y_pred)
    rscore = recall_score(y_test, y_pred)
    fscore = f1_score(y_test, y_pred)

    SF = 4
    output = f"KNN --> accuracy: {round(ascore,SF)}, precision: {round(pscore,SF)}, recall: {round(rscore,SF)}, f1 score: {round(fscore, SF)}"
    return columns, output, [ascore, pscore, rscore, fscore]
