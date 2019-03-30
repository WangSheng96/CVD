# this shows the number of cases where the same X leads to different Ys, sorted by score

import pandas as pd

data = pd.read_csv("data/new_cardio.csv").values

d = {}; r = lambda x:round(x,1)
for row in data:
    x,y = tuple([r(i) for i in row[:-1]]), int(row[-1])
    if x not in d.keys():
        d[x] = {"negative class":0, "positive class":0}
        d[x][["negative class","positive class"][y]] += 1
    else:
        d[x][["negative class","positive class"][y]] += 1

d = [(k,d[k]) for k in d]

def score(dic):
    a,b = dic["negative class"], dic["positive class"]
    return (1+abs(a-b))/(a+b)

d.sort(key=lambda x:score(x[1]))

print("\n\n")
for a,b in d[:40]:
    print(f"configuration of x: {a} -->\t{b}")
print("\n\n")
