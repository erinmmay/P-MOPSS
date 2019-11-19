import numpy as np
from setup import *

import matplotlib.pyplot as plt

def bin_y_shift(y,binny):
    print ypixels, binny, ypixels/binny
    y_orig=np.copy(y)
    for yi in range(0,len(y)):
        if y[yi]<=ypixels/binny:
            y[yi]=y[yi]*binny
            #print 'CASE1: ', y_orig[yi], y[yi]
        elif y[yi]>ypixels/binny and y[yi]<=ypixels/binny+ygap:
            dy=y[yi]-ypixels/binny
            y[yi]=(y[yi]-dy)*binny+dy
            #print 'CASE2: ', y_orig[yi], y[yi]
        elif y[yi]>ypixels/binny+ygap:
            y[yi]=(y[yi]-ygap)*binny+ygap
            #print 'CASE3: ', y_orig[yi], y[yi]
            
    plt.figure(201,figsize=(8,6))
    plt.plot(y_orig,y,'.',color='black')
    plt.axhline(y=ypixels)
    plt.axhline(y=ypixels+ygap)
    plt.axvline(x=ypixels/binny)
    plt.axvline(x=ypixels/binny+ygap)
    plt.show(block=False)
    return y
        