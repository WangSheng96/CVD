import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from os.path import isfile, join
from os import listdir

i = 0
for filename in listdir("data"):
    try:
        data = pd.read_csv("data/"+filename)
        # data = pd.read_csv("data/t1.csv")
        x,y = data.iloc[:,:-1], data.iloc[:,-1]

        pca = PCA()
        pca.fit(x)

        # print("explained_variance_ratio:",pca.explained_variance_ratio_)
        print(filename,"total of explained_variance_ratio:",sum(pca.explained_variance_ratio_))
        ################3

        print("components",pca.components_)
        print("explained variance raito:",pca.explained_variance_ratio_)




        data2 = pca.fit_transform(data)
        data2 = pd.DataFrame(data2)
        x1,x2 = data2.iloc[:,0], data2.iloc[:,1]

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

        def fun1(temp):
            return temp

        def fun2(temp):
            return temp

        px,py = [fun1(i[0]) for i in pos], [fun2(i[1]) for i in pos]
        nx,ny = [fun1(i[0]) for i in neg], [fun2(i[1]) for i in neg]

        plt.scatter(px,py,color="red")
        plt.scatter(nx,ny, color="green")
        plt.show()

        # i += 1
        # print(filename)
        # if i >= 2:
        #     break

    except:
        print(filename,"ERROR")

    break

