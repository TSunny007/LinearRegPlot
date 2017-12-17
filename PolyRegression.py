import math
import matplotlib as mpl

mpl.use('PDF')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from numpy import linalg as LA
from numpy import polyval
from scipy import polyfit
import random

# This plots our results appropriately
def plot_poly(x,y,xE,yE,p): 
    plt.scatter(x,y, s=30, c="blue") # training data
    plt.scatter(xE,yE, s=20, c="green") # testing data
    plt.axis([0,1.5,0,3])
    s=sp.linspace(0,10,101)
    coefs=sp.polyfit(x,y,p)
    ffit = np.poly1d(coefs)
    plt.plot(s,ffit(s),'r-',linewidth=2.0)
    resid = ffit(xE)
    RMSE = LA.norm(resid-yE)
    SSE= RMSE * RMSE
    title = "degree %s fit | SSE %0.3f" % (p, SSE) 
    plt.title(title)
    file = "CVpolyReg-%s.pdf" % p
    plt.savefig(file, bbox_inches='tight') 
    plt.clf()
    plt.cla()

df = pd.read_csv('D3.csv', names = ['x0','x1','x2','y'])
# the first column
xColumn = df['x0']
# the fourth column
yColumn = df['y']

xTrain = []
yTrain = []
xTest = []
yTest = []

TEST_SIZE = 10 #the size of our test size

rands = set()
# our testing data consists of 10 points (~10% of data provided)
while len(rands) < TEST_SIZE:
    newRand = random.randint(0,98)
    if newRand not in rands:
        rands.add(newRand)
randsCopy = rands.copy()
for x in range(0, 10, 1):
    d = rands.pop()
    xTest.append(xColumn[d])
    yTest.append(yColumn[d])
for x in range(0, len(xColumn), 1):
    if x not in randsCopy:
        xTrain.append(xColumn[x])
        yTrain.append(yColumn[x])
    # after this we will have two disjoint datasets 
    # xTest and yTest go together while xTrain and yTrain go together
    # we will be going through five degrees
p_vals = [1,2,3,4,5]
for i in p_vals:
    plot_poly(np.asarray(xTrain),np.asarray(yTrain), 
    np.asarray(xTest), np.asarray(yTest), i)
            