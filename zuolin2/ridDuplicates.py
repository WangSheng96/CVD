head = "age,height,weight,ap_hi,ap_lo,cholesterol_0,cholesterol_1,gluc_0,gluc_1,gender,smoke,alco,active,cardio"

data = []
mapUnclean = {}
with open("data/new_cardio.csv") as f:
    for line in f:
        line = line.strip().split(",")
        x1, y = line[:-1], int(line[-1])
        x = tuple([round(float(i),1) for i in x1])
        mapUnclean[x] = x1
        data.append((tuple(x),y))

d = {}
for x,y in data:
    if x not in d:
        d[x] = [0,0]
        d[x][y] += 1
    else:
        d[x][y] += 1

def score(lis):
    a,b = lis
    return (abs(a-b)+1)/(a+b)

d = [(k,d[k],score(d[k])) for k in d]
d.sort(key=lambda x:x[2])

[print(i) for i in d[:5]]

def getY(lis):
    return sorted([(i,lis[i]) for i in range(2)], key=lambda x:x[1])[-1][0]

print()
data = []
for x,y,z in d:
    y = getY(y) 
    string = ",".join((mapUnclean[x]+[str(y)]))
    data.append(string)

with open("lzlclean.csv","w") as f:
    f.write(head + "\n")
    for d in data:
        f.write(d+"\n")



