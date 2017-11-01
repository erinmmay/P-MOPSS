## ------------------------------------------------------------- ##
##  Next we read in our data (applying masks) and                ##
##   subtract darks, divide flats                                ##
## ------------------------------------------------------------- ##
import numpy as np
np.seterr('ignore')
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

def Extract2D(path,ex,SAVEPATH,binn,Lflat):
    print' -->> Loading Masks'
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    
    print' -->> Loading Flats'
    if binn!=1:
        flat=np.load(SAVEPATH+'binned_flat.npz')['flat']
    else:
        flat=(np.load(SAVEPATH+'Flats.npz')['medfilt'])
        flat_full=FullFrame(1,flat,1)[0,:,:]
        flat_full/=np.nanmedian(flat_full)
        print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
    
        #print '               (finding bad pixels in flat...)'
        #for i in range(0,flat_full.shape[0]):
        #    for j in range(0,flat_full.shape[1]):
        #        if flat_full[i,j]>1.4 or flat_full[i,j]<0.6:
#            ran=2
#            mini=np.max([0,i-ran])
#            maxi=np.min([i+ran,flat_full.shape[0]])
#            minj=np.max([0,j-ran])
#            maxj=np.min([j+ran,flat_full.shape[1]])
#            flat_full[0,i,j]=np.nanmedian(flat_full[0,mini:maxi,minj:maxj])
         #           flat_full[i,j]=np.nan
    
        del flat
    print'          (done)'
    

    print' -->> Loading Darks'
    dark=np.load(SAVEPATH+'Darks.npz')['medfilt']
    if Lflat==True:
        dark_full=FullFrame(1,dark,binn)[0,:,:]/flat_full
    else:
        dark_full=FullFrame(1,dark,binn)
    dark_med=np.nanmedian(dark_full)
    print '             ', dark_med                               #checking dark level
    del dark
    del dark_full
    print'          (done)'


    print' -->> Loading HeaderData'
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    print'              ', n_exp
    print'          (done)'
    
    data={}
    for i in range(0,masks.shape[0]):
        y0=np.int(masks[i,1])
        ywid=(np.int(masks[i,3]-masks[i,1]))
        #print y0, ywid, x0, xwid
        lowy=np.int(np.max([0,y0-ex]))
        topy=np.int(np.min([2*ypixels/binn+ygap, y0+ywid+ex]))
        data['obj'+str(int(i))]=np.empty([n_exp,np.int(topy-lowy),np.int(np.abs(masks[i,0]-masks[i,2]))])
    #data=np.empty([n_obj,n_exp,2*ypixels+ygap,200])*0.0
    
    data_2c=np.empty([2*ypixels/binn+ygap,xpixels/binn])*0.0
    exp_cnt=0

    image_full=np.empty([2*ypixels/binn+ygap,4*xpixels/binn+3*xgap])*0.0
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            split=file.split('c')
            root=file.split('c')[0]
            chip=int(file.split('c')[1].split('.')[0])
            #print split, root, chip
            if chip in top_chip:
                c=top_chip.index(chip)
                #    print '  -->>', root, chip, bot_chip[c]
                data_t=np.fliplr((fits.open(path+root+'c'+str(int(chip))+'.fits.gz')[0].data)[0:ypixels/binn,0:xpixels/binn])
                data_b=np.flipud((fits.open(path+root+'c'+str(int(bot_chip[c]))+'.fits.gz')[0].data)[0:ypixels/binn,0:xpixels/binn])
                #print root, c, np.nanmedian(data_t), np.nanmedian(data_b)
                data_2c[0:ypixels/binn,:]=data_t
                data_2c[ypixels/binn+ygap:,:]=data_b
                del data_t
                del data_b
                if c==0:
                    image_full[:,0:xpixels/binn]=data_2c
                if c==1:
                    image_full[:,xpixels/binn+xgap:2*xpixels/binn+xgap]=data_2c
                if c==2:
                    image_full[:,2*xpixels/binn+2*xgap:3*xpixels/binn+2*xgap]=data_2c
                if c==3:
                    image_full[:,3*xpixels/binn+3*xgap:]=data_2c
            exp_cnt+=(1./8.)
            if exp_cnt%1==0:
                #print '  -->> EXPOSURE # ', np.int(exp_cnt), np.nanmedian(image_full)
                if Lflat==True:
                    image_full/=flat_full[:,:]
                image_full-=dark_med
                #print '                     ', np.nanmedian(image_full)
                for i in range(0,n_obj):
                    x0=np.int(masks[i,0])
                    y0=np.int(masks[i,1])
                    xwid=(np.int(masks[i,2]-masks[i,0]))
                    ywid=(np.int(masks[i,3]-masks[i,1]))
                    #print y0, ywid, x0, xwid
                    lowy=np.int(np.max([0,y0-ex]))
                    topy=np.int(np.min([2*ypixels/binn+ygap, y0+ywid+ex]))
                    lowx=np.int(np.max([0,x0]))
                    topx=np.int(np.min([4*xpixels/binn+3*xgap,x0+xwid]))
                    (data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:]=image_full[lowy:topy,lowx:topx]
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
        np.savez_compressed(SAVEPATH+'2DSpec_obj'+str(int(k))+'.npz', data=save)
    return data



