## ------------------------------------------------------------- ##
##  Next we read in our data (applying masks) and                ##
##   subtract darks, divide flats                                ##
## ------------------------------------------------------------- ##
import numpy as np
import os

from astropy.io import fits
from setup import *

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

    print' -->> Loading Darks'
    dark=np.load('SaveData/Darks.npz')['medfilt']
    dark_full=FullFrame(1,dark)
    print '             ', np.nanmedian(dark_full)                   #checking dark level
    del dark
    print'          (done)'

    print' -->> Loading Flats'
    flat=np.load('SaveData/Darks.npz')['medfilt']
    flat_full=FullFrame(1,flat)
    flat_full/=np.nanmedian(flat_full)
    print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
    del flat
    print'          (done)'

    print' -->> Loading HeaderData'
    n_exp=np.load('SaveData/HeaderData.npz')['n_exp']
    print'              ', n_exp
    print'          (done)'
    
    data={}
    for i in range(0,masks.shape[0]):
        data['obj'+str(int(i))]=np.empty([n_exp,2*ypixels+ygap,np.int(np.abs(masks[i,0]-masks[i,2]))])

    data_2c=np.empty([2*ypixels+ygap,xpixels])*0.0
    exp_cnt=0

    image_full=np.empty([2*ypixels+ygap,4*xpixels+3*xgap])*np.nan
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
                image_full-=dark_full[0,:,:]
                image_full/=flat_full[0,:,:]
                #print ' -->> EXPOSURE # ', np.int(exp_cnt)
                for i in range(0,n_obj):
                    x0=np.int(masks[i,0])
                    y0=np.int(masks[i,1])
                    xwid=(np.int(masks[i,2]-masks[i,0]))
                    ywid=(np.int(masks[i,3]-masks[i,1]))
                    #print y0, ywid, x0, xwid
                    (data['obj'+str(int(i))])[np.int(n_exp)-1,y0:y0+ywid,:]=image_full[y0:y0+ywid,x0:x0+xwid]
                    del image_full
                    image_full=np.empty([2*ypixels+ygap,4*xpixels+3*xgap])*np.nan
                #if exp_cnt%10==0:
                print '           ( SAVED DATA FOR IMAGE ', np.int(exp_cnt), ')'
    np.savez('SaveData/2DSpec.npz',data=data)
    return data


