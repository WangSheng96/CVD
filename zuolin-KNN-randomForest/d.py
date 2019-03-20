import pickle

data = pickle.load(open("NNresults.sav","rb"))
data.sort(key=lambda x:x[1], reverse=True)


for a,d in data[:10]:
    print(a,d)


