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

   
def DetermineSide(nflats,path,ex,exx,SAVEPATH):
    print' -->> Loading Masks'
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    
#    print' -->> Loading Flats'
#    flat=np.load(SAVEPATH+'Darks.npz')['medfilt']
#    flat_full=FullFrame(1,flat)
#    flat_full/=np.nanmedian(flat_full)
#    print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
#    del flat
#    print'          (done)'
    

#    print' -->> Loading Darks'
#    dark=np.load(SAVEPATH+'Darks.npz')['medfilt']
#    dark_full=FullFrame(1,dark)/flat_full
#    dark_med=np.nanmedian(dark_full)
#    print '             ', dark_med                               #checking dark level
#    del dark
#    del dark_full
#    print'          (done)'


#    print' -->> Loading HeaderData'
#    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
#    print'              ', n_exp
#    print'          (done)'
    
    data={}
    for i in range(0,masks.shape[0]):
        y0=np.int(masks[i,1])
        ywid=(np.int(masks[i,3]-masks[i,1]))
        #print y0, ywid, x0, xwid
        lowy=np.int(np.max([0,y0-ex]))
        topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
        data['obj'+str(int(i))]=np.empty([nflats,np.int(topy-lowy),np.int(np.abs(masks[i,0]-masks[i,2]))+2*exx])
    #data=np.empty([n_obj,n_exp,2*ypixels+ygap,200])*0.0
    
    data_2c=np.empty([2*ypixels+ygap,xpixels])*0.0
    exp_cnt=0

    image_full=np.empty([2*ypixels+ygap,4*xpixels+3*xgap])*0.0
    while exp_cnt<=1:
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
                if exp_cnt==1:
                    for i in range(0,n_obj):
                        x0=np.int(masks[i,0])
                        y0=np.int(masks[i,1])
                        xwid=(np.int(masks[i,2]-masks[i,0]))
                        ywid=(np.int(masks[i,3]-masks[i,1]))
                        #print y0, ywid, x0, xwid
                        lowy=np.int(np.max([0,y0-ex]))
                        topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
                        lowx=np.int(np.max([0,x0-exx]))
                        topx=np.int(np.min([4*xpixels+3*xgap,x0+xwid+exx]))
                        (data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:]=image_full[lowy:topy,lowx:topx]
                        plt.clf()
                        plt.cla()
                        for y in range(0,topy-lowy):
                            if y%5==0:
                                plt.plot(np.linspace(lowx,topx,topx-lowx),data['obj'+str(int(i))][0,y,:])
                        plt.title('obj'+str(int(i)))
                        plt.show(block=False)
                        
    return
                   
def Extract_wave_left(path,ex,exx,SAVEPATH,obj,n_wave):
    n_exp=n_wave
    print' -->> Loading Masks'
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    
#     print' -->> Loading Flats'
#     flat=np.load(SAVEPATH+'Flats.npz')['medfilt']
#     flat_full=FullFrame(1,flat,1)
#     flat_full/=np.nanmedian(flat_full)
#     print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
#     del flat
#     print'          (done)'
    

#     print' -->> Loading Darks'
#     dark=np.load(SAVEPATH+'Darks.npz')['medfilt']
#     dark_full=FullFrame(1,dark,1)/flat_full
#     dark_med=np.nanmedian(dark_full)
#     print '             ', dark_med                               #checking dark level
#     del dark
#     del dark_full
#     print'          (done)'


#    print' -->> Loading HeaderData'
#    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
#    print'              ', n_exp
#    print'          (done)'
    
    data={}
    y0=np.int(masks[obj,1])
    ywid=(np.int(masks[obj,3]-masks[obj,1]))
    lowy=np.int(np.max([0,y0-ex]))
    topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
    summ=np.empty([n_exp,2*ypixels+ygap])*np.nan
        #data['obj'+str(int(i))]=np.empty([n_exp,np.int(topy-lowy),np.int(np.abs(masks[i,0]-masks[i,2]))+2*exx])
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
                print '  -->> EXPOSURE # ', np.int(exp_cnt)
#                 image_full/=flat_full[0,:,:]
#                 image_full-=dark_med
                #print '                     ', np.nanmedian(image_full)
                x0=np.int(masks[obj,0])
                y0=np.int(masks[obj,1])
                xwid=(np.int(masks[obj,2]-masks[obj,0]))
                ywid=(np.int(masks[obj,3]-masks[obj,1]))
                #print y0, ywid, x0, xwid
                lowy=np.int(np.max([0,y0-ex]))
                topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
                lowx=np.int(np.max([0,x0-exx]))
                topx=np.int(np.min([4*xpixels+3*xgap,x0+xwid+exx]))
                xcent=np.int(x0+xwid/2.)
                for p in range(0,topy-lowy):
                    summ[np.int(exp_cnt)-1,p+lowy]=np.nansum(image_full[p+lowy,lowx+10:x0-10])
#                     fig,ax=plt.subplots(1,2,figsize=(2.,4.))
#                     ax[0].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:],cmap=plt.cm.Greys_r)
#                     ax[1].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-2,:,:],cmap=plt.cm.Greys_r)
#                     plt.figtext(0.1,0.9,str(int(exp_cnt))+' '+str(int(i)),color='red')
#                     plt.show(block=False)
#                     plt.close()
            #if exp_cnt%10==0:
            #    print '           ( EXTRACTED DATA FOR IMAGE ', np.int(exp_cnt), ')  --   ', n_exp, ' exposures total'
    for s in range(0,summ.shape[1]):
        for t in range(0,n_exp):
            if summ[t,s]<0 or summ[t,s]>5*10**5.:
                summ[t,s]=np.nan
    summ=np.nanmedian(summ,axis=0)
    plt.plot(summ)
    plt.show(block=False)
    np.savez_compressed(SAVEPATH+'wavespec_obj'+str(int(obj))+'.npz',spec=np.flip(summ,0))
    return summ

def Extract_wave_right(path,ex,exx,SAVEPATH,obj,n_wave):
    n_exp=n_wave
    print' -->> Loading Masks'
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    
#     print' -->> Loading Flats'
#     flat=np.load(SAVEPATH+'Flats.npz')['medfilt']
#     flat_full=FullFrame(1,flat,1)
#     flat_full/=np.nanmedian(flat_full)
#     print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
#     del flat
#     print'          (done)'
    

#     print' -->> Loading Darks'
#     dark=np.load(SAVEPATH+'Darks.npz')['medfilt']
#     dark_full=FullFrame(1,dark,1)/flat_full
#     dark_med=np.nanmedian(dark_full)
#     print '             ', dark_med                               #checking dark level
#     del dark
#     del dark_full
#     print'          (done)'


#   print' -->> Loading HeaderData'
#   n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
#   print'              ', n_exp
#   print'          (done)'
    
    data={}
    y0=np.int(masks[obj,1])
    ywid=(np.int(masks[obj,3]-masks[obj,1]))
        #print y0, ywid, x0, xwid
    lowy=np.int(np.max([0,y0-ex]))
    topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
    summ=np.empty([n_exp,2*ypixels+ygap])*np.nan
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
                print '  -->> EXPOSURE # ', np.int(exp_cnt)
#                 image_full/=flat_full[0,:,:]
#                 image_full-=dark_med
                #print '                     ', np.nanmedian(image_full)
                for i in range(0,n_obj):
                    x0=np.int(masks[obj,0])
                    y0=np.int(masks[obj,1])
                    xwid=(np.int(masks[obj,2]-masks[obj,0]))
                    ywid=(np.int(masks[obj,3]-masks[obj,1]))
                    #print y0, ywid, x0, xwid
                    lowy=np.int(np.max([0,y0-ex]))
                    topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
                    lowx=np.int(np.max([0,x0-exx]))
                    topx=np.int(np.min([4*xpixels+3*xgap,x0+xwid+exx]))
                    for p in range(0,topy-lowy):
                        summ[np.int(exp_cnt)-1,p+lowy]=np.nansum(image_full[p+lowy,x0+xwid+10:topx-10])
#                     fig,ax=plt.subplots(1,2,figsize=(2.,4.))
#                     ax[0].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:],cmap=plt.cm.Greys_r)
#                     ax[1].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-2,:,:],cmap=plt.cm.Greys_r)
#                     plt.figtext(0.1,0.9,str(int(exp_cnt))+' '+str(int(i)),color='red')
#                     plt.show(block=False)
#                     plt.close()
            #if exp_cnt%10==0:
            #    print '           ( EXTRACTED DATA FOR IMAGE ', np.int(exp_cnt), ')  --   ', n_exp, ' exposures total'
    for s in range(0,summ.shape[1]):
        for t in range(0,n_exp):
            if summ[t,s]<0 or summ[t,s]>5*10**5.:
                summ[t,s]=np.nan
    summ=np.nanmedian(summ,axis=0)
    plt.plot(summ)
    plt.show(block=False)
    np.savez_compressed(SAVEPATH+'wavespec_obj'+str(int(obj))+'.npz', spec=np.flip(summ,0))
    return summ


def Extract_wave_center(path,ex,exx,SAVEPATH,obj,n_wave):
    n_exp=n_wave
    print' -->> Loading Masks'
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    
#     print' -->> Loading Flats'
#     flat=np.load(SAVEPATH+'Flats.npz')['medfilt']
#     flat_full=FullFrame(1,flat,1)
#     flat_full/=np.nanmedian(flat_full)
#     print '             ', np.nanmedian(flat_full)                   #checking that flat has been normalized to 1
#     del flat
#     print'          (done)'
    

#     print' -->> Loading Darks'
#     dark=np.load(SAVEPATH+'Darks.npz')['medfilt']
#     dark_full=FullFrame(1,dark,1)/flat_full
#     dark_med=np.nanmedian(dark_full)
#     print '             ', dark_med                               #checking dark level
#     del dark
#     del dark_full
#     print'          (done)'


#   print' -->> Loading HeaderData'
#   n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
#   print'              ', n_exp
#   print'          (done)'
    
    data={}
    y0=np.int(masks[obj,1])
    ywid=(np.int(masks[obj,3]-masks[obj,1]))
        #print y0, ywid, x0, xwid
    lowy=np.int(np.max([0,y0-ex]))
    topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
    summ=np.empty([n_exp,2*ypixels+ygap])*np.nan
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
                print '  -->> EXPOSURE # ', np.int(exp_cnt)
#                 image_full/=flat_full[0,:,:]
#                 image_full-=dark_med
                #print '                     ', np.nanmedian(image_full)
                for i in range(0,n_obj):
                    x0=np.int(masks[obj,0])
                    y0=np.int(masks[obj,1])
                    xwid=(np.int(masks[obj,2]-masks[obj,0]))
                    ywid=(np.int(masks[obj,3]-masks[obj,1]))
                    #print y0, ywid, x0, xwid
                    lowy=np.int(np.max([0,y0-ex]))
                    topy=np.int(np.min([2*ypixels+ygap, y0+ywid+ex]))
                    lowx=np.int(np.max([0,x0-exx]))
                    topx=np.int(np.min([4*xpixels+3*xgap,x0+xwid+exx]))
                    xcent=np.int(x0+xwid/2.)
                    for p in range(0,topy-lowy):
                        summ[np.int(exp_cnt)-1,p+lowy]=np.nansum(image_full[p+lowy,xcent-20:xcent+20])
                    #fig,ax=plt.subplots(1,2,figsize=(2.,4.))
                    #ax[0].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:],cmap=plt.cm.Greys_r)
                    #ax[1].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-2,:,:],cmap=plt.cm.Greys_r)
                    #plt.figtext(0.1,0.9,str(int(exp_cnt))+' '+str(int(i)),color='red')
                    #plt.show(block=False)
            #if exp_cnt%10==0:
            #    print '           ( EXTRACTED DATA FOR IMAGE ', np.int(exp_cnt), ')  --   ', n_exp, ' exposures total'
    for s in range(0,summ.shape[1]):
        for t in range(0,n_exp):
            if summ[t,s]<0 or summ[t,s]>5*10**5.:
                summ[t,s]=np.nan
    summ=np.nanmedian(summ,axis=0)
    
    plt.plot(summ)
    plt.show(block=False)
    #np.savez(SAVEPATH+'wavespec_obj'+str(int(obj))+'.npz', spec=np.flip(summ,0))
    return summ