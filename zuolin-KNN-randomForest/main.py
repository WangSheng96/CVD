# knn
#     num_neighbours = 5
#     algorithm = auto, ball_tree, brute, kd_tree
#     leaf_size = 30
#     P = 2

# randomForest
#     n_estimators = 10        
#     criterion = "gini"      
#     max_depth = None            
from knn import *
from randomForest import *
from os import listdir
from os.path import isfile, join

ERRORS = [] 

knnd = [0,0,0,0]
rfd = [0,0,0,0]

ii = 0
with open("report.txt","w") as f:
    for filename in listdir("data"):
        try:
            f.write(filename + ":\n") 
            filename = "data/" + filename
            columns, knnResults, kscores = knn(filename)
            rfResults, rscores = randomForest(filename)
            
            for i in range(4):
                if kscores[i] > knnd[i]:
                    knnd[i] = kscores[i]
                if rscores[i] > rfd[i]:
                    rfd[i] = rscores[i]
            # f.write(columns + "\n" + knnResults + "\n" + rfResults + "\n\n\n")
        except:
            ERRORS.append(filename)
        # ii += 1
        # if ii == 5:
        #     break

print("num errors:",len(ERRORS))

labels = ["accuracy","precision","recall","f1 score"]

print("k nearest neighbours")
for i in range(4):
    print(labels[i] + ": " + str(knnd[i]), end=",  ")

print("\n")
print("random forest")
for i in range(4):
    print(labels[i] + ": " + str(rfd[i]), end=",  ")

