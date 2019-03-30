import pandas as pd
import numpy as np

data = pd.read_csv("new_cardio.csv")

d = {}

# TUNING PARAMETERS HERE

DECIMAL_PLACE = 2

# END OF TUNING PARAMETERS

for row in data.values:
    x,y = row[:-1], int(row[-1])
    x = tuple([round(i,DECIMAL_PLACE) for i in x])

    if x not in d.keys():
        d[x] = [0,0]
        d[x][y] += 1
    else:
        d[x][y] += 1

d = [(k,d[k]) for k in d]

def score(tup):
    a,b = tup
    return (1+abs(a-b))/(a+b)
d.sort(key=lambda x:score(x[1]))

for a,b in d[:5]:
    print(a,b)







# WRITING TO NEW DATABASE
# FILENAME = f"data/cardio_{DECIMAL_PLACE}dp.csv"
# with open(FILENAME,"w") as f:
#     f.write(",".join(data.columns)+"\n")
#     for k in d:
#         a,b = d[k]
#         score = str(b/(a+b))
#         f.write(",".join([str(i) for i in k])+","+score+"\n")
        
