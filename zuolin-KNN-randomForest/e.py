data = []
with open("data/new_cardio.csv") as f:
    for line in f:
        data.append(line.strip().lower().split(","))
labels = data.pop(0)
# print(" ".join(labels))

data2 = []
for row in data:
    x,y = row[:-1], row[-1]
    data2.append((tuple([round(float(i),1) for i in x]), float(y)))


# for i,(x,y) in enumerate(data2[:5]):
#     print(x,y)

d = {}
for i,(x,y) in enumerate(data2):
    print("   ",end="\r")

    if x not in d:
        d[x] = [0,0]
        d[x][int(y)] += 1
    
    else:
        d[x][int(y)] += 1

print(len(d))

# ONLY 14293 UNIQUE ENTRIES BASED ON X

def compute(t):
    a,b = t
    return (abs(a-b))/(a+b)-(a+b)/2


values = list(d.values())
output = []
for v in values:
    score = compute(v)
    output.append((v,score))

output.sort(key= lambda x:x[1])

for (a,b),c in output[:30]:
    print((a,b),"   ",(abs(a-b)/(a+b)))
 
 




