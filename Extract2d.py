## ------------------------------------------------------------- ##
##  Next we read in our data (applying masks) and                ##
##   subtract darks, divide flats                                ##
## ------------------------------------------------------------- ##
import numpy as np
import os

import cPickle as pickle

from astropy.io import fits
from setup import *

import matplotlib.pyplot as plt

top_chip=[6,5,8,7]
bot_chip=[1,2,3,4]

#path=data_path
#print path

from FullFrame import FullFrame

def Extract2D(path):
    print' -->> Loading Masks'
    masks=np.load('SaveData/FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    
    print' -->> Loading Flats'
    flat=np.load('SaveData/Darks.npz')['medfilt']
    flat_full=FullFrame(1,flat)
    flat_full/=np.nanmedian(flat_full)
    print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
    del flat
    print'          (done)'
    

    print' -->> Loading Darks'
    dark=np.load('SaveData/Darks.npz')['medfilt']
    dark_full=FullFrame(1,dark)/flat_full
    dark_med=np.nanmedian(dark_full)
    print '             ', dark_med                               #checking dark level
    del dark
    del dark_full
    print'          (done)'


    print' -->> Loading HeaderData'
    n_exp=np.load('SaveData/HeaderData.npz')['n_exp']
    print'              ', n_exp
    print'          (done)'
    
    data={}
    for i in range(0,masks.shape[0]):
        data['obj'+str(int(i))]=np.empty([n_exp,np.int(np.abs(masks[i,3]-masks[i,1])),np.int(np.abs(masks[i,0]-masks[i,2]))])
    #data=np.empty([n_obj,n_exp,2*ypixels+ygap,200])*0.0
    
    data_2c=np.empty([2*ypixels+ygap,xpixels])*0.0
    exp_cnt=0

    image_full=np.empty([2*ypixels+ygap,4*xpixels+3*xgap])*0.0
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            split=file.split('c')
            root=file.split('c')[0]
            chip=int(file.split('c')[1].split('.')[0])
            #print split, root, chip
            if chip in top_chip:
                c=top_chip.index(chip)
                #    print '  -->>', root, chip, bot_chip[c]
                data_t=np.fliplr((fits.open(path+root+'c'+str(int(chip))+'.fits.gz')[0].data)[0:ypixels,0:xpixels])
                data_b=np.flipud((fits.open(path+root+'c'+str(int(bot_chip[c]))+'.fits.gz')[0].data)[0:ypixels,0:xpixels])
                #print root, c, np.nanmedian(data_t), np.nanmedian(data_b)
                data_2c[0:ypixels,:]=data_t
                data_2c[ypixels+ygap:,:]=data_b
                del data_t
                del data_b
                if c==0:
                    image_full[:,0:xpixels]=data_2c
                if c==1:
                    image_full[:,xpixels+xgap:2*xpixels+xgap]=data_2c
                if c==2:
                    image_full[:,2*xpixels+2*xgap:3*xpixels+2*xgap]=data_2c
                if c==3:
                    image_full[:,3*xpixels+3*xgap:]=data_2c
            exp_cnt+=(1./8.)
            if exp_cnt%1==0:
                #print '  -->> EXPOSURE # ', np.int(exp_cnt), np.nanmedian(image_full)
                image_full/=flat_full[0,:,:]
                image_full-=dark_med
                #print '                     ', np.nanmedian(image_full)
                for i in range(0,n_obj):
                    x0=np.int(masks[i,0])
                    y0=np.int(masks[i,1])
                    xwid=(np.int(masks[i,2]-masks[i,0]))
                    ywid=(np.int(masks[i,3]-masks[i,1]))
                    #print y0, ywid, x0, xwid
                    (data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:]=image_full[y0:y0+ywid,x0:x0+xwid]
                    #fig,ax=plt.subplots(1,2,figsize=(2.,4.))
                    #ax[0].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:],cmap=plt.cm.Greys_r)
                    #ax[1].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-2,:,:],cmap=plt.cm.Greys_r)
                    #plt.figtext(0.1,0.9,str(int(exp_cnt))+' '+str(int(i)),color='red')
                    #plt.show(block=False)
            if exp_cnt%10==0:
                print '           ( EXTRACTED DATA FOR IMAGE ', np.int(exp_cnt), ')  --   ', n_exp, ' exposures total'
    for k in range(0,n_obj):
        save=data['obj'+str(int(k))]
        #print k, np.nanmedian(save)
        np.savez('SaveData/2DSpec_obj'+str(int(k))+'.npz', data=save)
    return data



