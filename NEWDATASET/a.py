import pickle

data = pickle.load(open("saved/results.sav","rb"))

data.sort(key=lambda x:x[1]["accuracy"])

for d in data:
    print(d)

