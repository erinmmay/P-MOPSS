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

top_gain=[0.82,0.83,0.82,0.86]
bot_gain=[0.85,0.85,0.84,0.83]

#path=data_path
#print path

from FullFrame import FullFrame
from FullFrame import Bin_FullFrame
from MasterFrame import MasterFrame

def Extract2D(path,ex,SAVEPATH,FLATPATH_SPEC,DARKPATH,binnx,binny,fb,Lflat,Ldark):
    print' -->> Loading Masks'
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']
    print'          (done)'

    n_obj=int(masks.shape[0])
    print' -->> Loading HeaderData'
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    print'              ', n_exp
    print'          (done)'
    print ' ' 
    
    
    dark_vars=np.empty([n_exp])*np.nan
    if Ldark==True:
        print' ->> LOADING DARKS'
        exp_times=np.load(SAVEPATH+'HeaderData.npz')['exp_times']
        ###############
        expt0=exp_times[int(n_exp/2)]
        dark0=dark=MasterFrame(DARKPATH+'_'+str(int(expt0))+'/','Darks',SAVEPATH,binnx,binny) 
        dark_full0=FullFrame(1,dark0,binnx,binny)
        del dark0
        dark_med=np.nanmedian(dark_full0)
        del dark_full0
        ###################
        for t in range(0,n_exp):
            #print '      exposure', t, exp_times[t],expt0
            if exp_times[t]==expt0:
                dark_vars[t]=dark_med
            else:
                print '      exposure', t, ': DIFFERENT exp_time!'
                expt=exp_times[t]
                darkt=dark=MasterFrame(DARKPATH+'_'+str(int(expt))+'/','Darks',SAVEPATH,binnx,binny) 
                dark_fullt=FullFrame(1,darkt,binnx,binny) 
                del darkt
                dark_vars[t]=np.nanmedian(dark_fullt)
                del dark_fullt
               
                
        print '             ', dark_vars                              #checking dark level
        print'          (done)'
        print ' '
    
    if Lflat==True:
        print' ->> LOADING FLATS'

        flatspec=MasterFrame(FLATPATH_SPEC,'Flats',SAVEPATH,fb,fb)
        if fb==binnx and fb==binny:
            print '      same binning'
            flat_spec=FullFrame(1,flatspec,fb,fb)[0,:,:]
#             plt.figure(102)
#             plt.imshow(flat_spec)
#             plt.show(block=False)
            #flat_spec/=np.nanmedian(flat_spec)
        elif fb!=binnx or fb!=binny:
            print '      adjusting binning'
            flat_spec=Bin_FullFrame(1,flatspec,binnx,binny)[0,:,:]
            #flat_full/=np.nanmedian(flat_full)
            #flat spec later separated into each object, a curve fit and then removed, then divided out

        #print '             ', np.nanmedian(flat_full)                   #checking that full flat has been normalized to 1

        #del flat
        print'          (done)'
        print ' '

    data={}
    FLAT={}
    
    x0_arr=np.empty(masks.shape[0])*np.nan
    y0_arr=np.empty(masks.shape[0])*np.nan
    xwid_arr=np.empty(masks.shape[0])*np.nan
    ywid_arr=np.empty(masks.shape[0])*np.nan
    
    lowx_arr=np.empty(masks.shape[0])*np.nan
    topx_arr=np.empty(masks.shape[0])*np.nan
    
    lowy_arr=np.empty(masks.shape[0])*np.nan
    topy_arr=np.empty(masks.shape[0])*np.nan
    
    for i in range(0,masks.shape[0]):
        x0=np.int(masks[i,0])
        y0=np.int(masks[i,1])
        xwid=(np.int(masks[i,2]-masks[i,0]))
        ywid=(np.int(masks[i,3]-masks[i,1]))
        ### BINN THE SIZES OF THE MASKS in Y ###
        if fb==1:
            if y0<=ypixels:
                y0=y0/binny
            if y0>ypixels and y0<=ypixels+ygap:
                dy=y0-ypixels
                y0=(y0-ypixels-dy)/binny+ypixels/binny+dy
            if y0>ypixels+ygap:
                y0=(y0-ypixels-ygap)/binny+ypixels/binny+ygap
            if ywid<=ypixels:
                ywid=ywid/binny
            if ywid>ypixels and ywid<=ypixels+ygap:
                dy=ywid-ypixels
                ywid=(ywid-dy)/binny+dy
            if ywid>ypixels:
                ywid=(ywid-ygap)/binny+ygap
            ########
            if x0<=xpixels:
                x0=x0/binnx
            if x0>xpixels and x0<=2*xpixels+xgap:
                x0=(x0-xpixels-xgap)/binnx+xpixels/binnx+xgap
            if x0>2*xpixels+xgap and x0<=3*xpixels+2*xgap:
                x0=(x0-2*xpixels-2*xgap)/binnx+2*xpixels/binnx+2*xgap
            if x0>3*xpixels+2*xgap and x0<=4*xpixels+3*xgap:
                x0=(x0-3*xpixels-3*xgap)/binnx+3*xpixels/binnx+3*xgap
            if xwid<xpixels:
                xwid=xwid/binnx
                
        lowx=np.int(np.max([0,x0]))
        topx=np.int(np.min([4*xpixels/binnx+3*xgap,x0+xwid]))
        #print y0, ywid, x0, xwid
        
        lowy=np.int(np.max([0,y0-ex]))
        topy=np.int(np.min([2*ypixels/binny+ygap, y0+ywid+ex]))
        
        x0_arr[i]=x0
        y0_arr[i]=y0
        
        xwid_arr[i]=xwid
        ywid_arr[i]=ywid
        
        lowx_arr[i]=lowx
        topx_arr[i]=topx
        
        lowy_arr[i]=lowy
        topy_arr[i]=topy
     
        data['obj'+str(int(i))]=np.empty([n_exp,np.int(topy-lowy),(xwid)])*np.nan
        if Lflat==True:
            print '  ****obj ', i
            fig,ax=plt.subplots(3,1,figsize=(15,9))
            for a in ax:
                a.set_xticklabels([])
            plt.subplots_adjust(wspace=0, hspace=0)
            
            ax[0].imshow(flat_spec[lowy:topy,lowx:topx].T,aspect='auto')
            
            flat_flat=np.nanmedian(flat_spec[lowy:topy,lowx+10/binnx:topx-10/binnx],axis=1)
            for j in range(0,xwid):
                flat_spec[lowy:topy,lowx+j]/=flat_flat  
            flat_spec[lowy:topy,lowx:topx]/=np.nanmedian(flat_spec[lowy:topy,lowx+10/binnx:topx-10/binnx])
            FLAT['obj'+str(int(i))]=flat_spec[lowy:topy,lowx:topx]
            
            ax[1].plot(flat_flat,linewidth=2.0,color='black')
            ax[2].imshow(flat_spec[lowy:topy,lowx:topx].T,aspect='auto')
            
            #fig.set_title('FLAT FOR OBJECT: '+str(int(i)))
            plt.figtext(0.2,0.3,np.nanmedian(flat_spec[lowy:topy,lowx+10/binnx:topx-10/binnx]),color='white',fontsize=20)
            plt.show(block=False)
              
    # create flat field model
        
    
    #data=np.empty([n_obj,n_exp,2*ypixels+ygap,200])*0.0
    
    data_2c=np.empty([2*ypixels/binny+ygap,xpixels/binnx])*np.nan
    exp_cnt=0

    image_full=np.empty([2*ypixels/binny+ygap,4*xpixels/binnx+3*xgap])*np.nan
    
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            split=file.split('c')
            root=file.split('c')[0]
            chip=int(file.split('c')[1].split('.')[0])
            #print split, root, chip
            if chip in top_chip:
                c=top_chip.index(chip)
                #    print '  -->>', root, chip, bot_chip[c]
                data_t=np.fliplr((fits.open(path+root+'c'+str(int(chip))+'.fits.gz')[0].data)
                                 [0:ypixels/binny,0:xpixels/binnx])/top_gain[c]
                data_b=np.flipud((fits.open(path+root+'c'+str(int(bot_chip[c]))+'.fits.gz')[0].data)
                                 [0:ypixels/binny,0:xpixels/binnx])/bot_gain[c]
                #print root, c, np.nanmedian(data_t), np.nanmedian(data_b)
                data_2c[0:ypixels/binny,:]=data_t
                data_2c[ypixels/binny+ygap:,:]=data_b
                del data_t
                del data_b
                if c==0:
                    image_full[:,0:xpixels/binnx]=data_2c
                if c==1:
                    image_full[:,xpixels/binnx+xgap:2*xpixels/binnx+xgap]=data_2c
                if c==2:
                    image_full[:,2*xpixels/binnx+2*xgap:3*xpixels/binnx+2*xgap]=data_2c
                if c==3:
                    image_full[:,3*xpixels/binnx+3*xgap:]=data_2c
            exp_cnt+=(1./8.)
            if exp_cnt%1==0:
                #print '  -->> EXPOSURE # ', np.int(exp_cnt), np.nanmedian(image_full)
                #print '                     ', np.nanmedian(image_full)
                for i in range(0,n_obj):
                    ly=int(lowy_arr[i])
                    ty=int(topy_arr[i])
                    lx=int(lowx_arr[i])
                    tx=int(topx_arr[i])
                    sd=image_full[ly:ty,lx:tx]
                    if Lflat==True:
                        sd/=FLAT['obj'+str(int(i))]
                    if Ldark==True:
                        sd-=dark_vars[i]
                    (data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:]=sd
#                     fig,ax=plt.subplots(1,2,figsize=(2.,4.))
#                     ax[0].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-1,:,:],cmap=plt.cm.Greys_r)
#                     ax[1].contourf((data['obj'+str(int(i))])[np.int(exp_cnt)-2,:,:],cmap=plt.cm.Greys_r)
#                     plt.figtext(0.1,0.9,str(int(exp_cnt))+' '+str(int(i)),color='red')
#                     plt.show(block=False)
            if exp_cnt%10==0:
                print '           ( EXTRACTED DATA FOR IMAGE ', np.int(exp_cnt), ')  --   ', n_exp, ' exposures total'
    for k in range(0,n_obj):   
        saved=data['obj'+str(int(k))]
        savef=np.nan
        if Lflat==True:
            savef=FLAT['obj'+str(int(k))]
        #print k, np.nanmedian(save)
        np.savez_compressed(SAVEPATH+'2DSpec_obj'+str(int(k))+'.npz', data=saved,flat=savef,dark=dark_vars[k])
    return data



