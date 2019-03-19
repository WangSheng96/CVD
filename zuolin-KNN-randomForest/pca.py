import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

data = pd.read_csv("data/new_cardio.csv")
x,y = data.iloc[:,:-1], data.iloc[:,-1]

pca = PCA(n_components=2)
pca.fit(x)

print("explained_variance_ratio:",pca.explained_variance_ratio_)
print("total of explained_variance_ratio:",sum(pca.explained_variance_ratio_))

data2 = pca.fit_transform(data)
data2 = pd.DataFrame(data2)
x1,x2 = data2.iloc[:,0], data2.iloc[:,1]

print(y.shape)
print(data2.shape)

plot = data2.values
y = y.values


pos = []
neg = []

for i in range(plot.shape[0]):
    a,b = int(y[i]), tuple(plot[i]) 
    if a == 0:
        neg.append(b)
    elif a == 1:
        pos.append(b)

from math import sin

def fun(temp):
    return sin(temp)

px,py = [sin(i[0]) for i in pos], [i[1] for i in pos]
nx,ny = [sin(i[0]) for i in neg], [i[1] for i in neg]

plt.scatter(px,py,color="red")
plt.scatter(nx,ny, color="blue")
plt.show()

