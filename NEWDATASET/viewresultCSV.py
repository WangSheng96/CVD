data = []
with open("results.csv") as f:
    for line in f:
        title, scores = line.strip().split("--> \t\t")
        scores = [tuple([float(i) if i[0]=="0" else i for i in score.strip().split(": ")]) for score in scores.strip().split(",")]


        data.append((title, scores))

# print(data[0])

data.sort(key=lambda x:x[1][0][1], reverse=True)

for a,b in data:
    print(a,b[0])


