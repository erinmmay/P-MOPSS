import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt

from datetime import datetime

from setup import *

def medcalc(obj_data,t,n_exp):
    if t>=2 and t<n_exp-2:
        dn2=obj_data[t-2]-obj_data[t]
        dn1=obj_data[t-1]-obj_data[t]
        up1=obj_data[t+1]-obj_data[t]
        up2=obj_data[t+2]-obj_data[t]
        med=np.nanmedian([dn2,dn1,up1,up2],axis=0)
    return med

def Outlier(obj,f,SAVEPATH):
    #n_obj=int(np.load('SaveData/FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    i=obj
    print '-----------------'
    print '  OBJECT # ', i
    print '-----------------'
    print '    --> Loading Data...'
    obj_data=(np.load(SAVEPATH+'2DSpec_obj'+str(int(i))+'.npz'))['data']
    ysize=obj_data.shape[1]
    xsize=obj_data.shape[2]
    # n_exp, y_pixs,x_pixs
    print '    --> Creating Diff Images...'
    diff=np.empty([obj_data.shape[0],obj_data.shape[1],obj_data.shape[2]])
    for t in range(2,n_exp-2):
        if (t+1)%10==0:
            print '         > EXPOSURE ', t+1
        diff[t,:,:]=medcalc(obj_data,t,n_exp)
    print  '    --> Calculating Medians...'
    window=8
    med_f=medfilt(diff[:,:,:],kernel_size=[1,1,window+1])
    win_s=np.empty([n_exp,ysize,xsize])
    for x in range(window,xsize-window):
        win_s[:,:,x]=np.std(diff[:,:,np.int(x-window):np.int(x+window)],axis=2)
    #med_r=np.nanmedian(diff[:,:,:],axis=2)
    #row_s=np.std(diff[:,:,:],axis=2)
    print  '    --> Identifying Outliers...'
    for t in range(2,n_exp-2): 
        if (t+1)%10==0:
            print '         > EXPOSURE ', t+1
        for y in range(0,ysize):
            for x in range(window,xsize-window):
                if np.abs(diff[t,y,x]-med_f[t,y,x])>f*win_s[t,y,x]:# or np.abs(diff[t,y,x]-med_r[t,y])>f*row_s[t,y]:
                    obj_data[t,y,x]=np.median(obj_data[t-2:t+2,y,x],0)
    np.savez(SAVEPATH+'Corrected'+str(int(i))+'.npz',data=obj_data)
    del obj_data
    #del diff
    #del med_f
    #del win_s
    #del med_r
    #del row_s
    #return obj_data
                
        
        
    
