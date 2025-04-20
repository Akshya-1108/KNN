import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sympy.integrals.intpoly import distance_to_side


def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt, k =5):
    m = X.shape[0]
    dlist = []
 
    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i]))
    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:,1]

    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]
    # print(dlist)
    # print(labels,cnts)
    return int(pred)


X,y = make_blobs(n_samples=2000,
                 n_features=2,
                 cluster_std= 3,
                 centers= 3,
                 random_state=42)
n_features = X.shape[1]
m = X.shape[0]

xt = np.array([-10,5])

for i in range(m):
    if y[i]== 0:
        plt.scatter(X[i,0], X[i,1], c = 'r', label= 'red')
    elif y[i]==1:
        plt.scatter(X[i,0], X[i,1], c = 'g', label= 'green')
    else:
        plt.scatter(X[i,0], X[i,1], c = 'b', label= 'blue')

plt.scatter(xt[0],xt[1], color = 'orange', marker = '*')


print(knn(X,y,xt))
# plt.show()