import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from datetime import datetime

from setup import *

from outlier_removal import outlierr_c

def BG_remove(extray,SAVEPATH,binnx,binny,Lflat,Ldark,ed_l,ed_u,ed_t,ks_b,sig_b,o_b,
                ver,ver_full,ver_t,trip,time_start,time_trim,obj_skip):
    #extray= number of pixels in y direction extra that were extracted
    #SAVEPATH= location of saved 2D spec
    #filename= name of saved file
    #savefile= name to save as
    #corr= cosmic ray corecction?
    #ed_l= location of lower boundary between background/data
    #ed_u= location of upper boundary between background/data
    #ed_t= number of pixels on edges of 2D strip to trim
    #ks_b= kernel size for background outlier detection
    #sig_b= sigma threshold for background outliers
    #o_b = order of polynomial for background fit
    #ver = verbose output (LOTS OF PLOTS!)
    #trip = Boolean. Run median filter 3 times
    
    ###########################################
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    
    bkgd_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,o_b+1])*np.nan
    
    if Ldark==True:
        dark_var=np.load(SAVEPATH+'Darks.npz')['var']
    if Lflat==True:
        flat_var=np.load(SAVEPATH+'Flats.npz')['var']
    
    for o in range(0,n_obj):
        if o in obj_skip:
            continue
        
        time0=datetime.now()
        print '-----------------'
        print '  OBJECT # ', o
        print '-----------------'
        
        print ' ----> loading data...'
        obj_data=(np.load(SAVEPATH+'2DSpec_obj'+str(int(o))+'.npz'))['data']
        print '      (done)'
        
        print ' ----> loading masks...'
        mask=(np.load(SAVEPATH+'FinalMasks.npz')['masks'])[o,:]
        print '      (done)'
        
        y0=int(mask[1])  #pixel number of inital extraction
        #y_start=np.int(np.max([0,y0-extray]))  #including extray
        ### BINN THE SIZES OF THE MASKS in Y ###
        if binny>1:
            if y0<=ypixels:
                y0=y0/binny
            if y0>ypixels and y0<=ypixels+ygap:
                dy=y0-ypixels
                y0=(y0-ypixels-dy)/binny+ypixels/binny+dy
            if y0>ypixels+ygap:
                y0=(y0-ypixels-ygap)/binny+ypixels/binny+ygap
             
        y_start=np.int(np.max([0,y0-extray]))  #including extray
#             if y_start<=ypixels:
#                 y_start=y_start/binny
#             if y_start>ypixels and y_start<=ypixels+ygap:
#                 dy=y_start-ypixels
#                 y_start=(y_start-ypixels-dy)/binny+ypixels/binny+dy
#             if y_start>ypixels+ygap:
#                 y_start=(y_start-ypixels-ygap)/binny+ypixels/binny+ygap
                
        ##################################
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        print o, n_rows,y_start
        
        xpix_ar=np.linspace(1,xwidth,xwidth)
        #fwhm_av=np.empty([n_exp])*np.nan
        
        bkgd_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        corr_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
    
        sub_bkgd=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        
        #begin loop over time...
        for t in range(time_start,n_exp-time_trim):
            if t%10==0:
                print '       *** TIME: ',t,' ***'
            
            frame=np.copy(obj_data[t,:,:])   #current frame
            ################# VERBOSE OUTPUT #################
            if ver_full==True:
                plt.figure(102,figsize=(14,4))
                plt.title('RAW: OBJ='+str(int(o))+' TIME='+str(int(t)))
                for j in range(0,n_rows):
                    plt.plot(xpix_ar,frame[j,:],linewidth=1.0)
                plt.axvline(x=ed_l, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
                plt.axvline(x=ed_t, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_t,color='grey',linewidth=0.5)
                plt.show(block=False)
                plt.close()
            ###################################################
            
            for j in range(0,n_rows):
                row_data=np.copy(frame[j,:])
                
                bg_pix=np.append(xpix_ar[ed_t:ed_l],xpix_ar[xwidth-ed_u:xwidth-ed_t])
                bg_dat_0=np.array(np.append(row_data[ed_t:ed_l],row_data[xwidth-ed_u:xwidth-ed_t]))
                
                ### do median filter ###
                c1=0
                c2=0
                c3=0
                c1,bg_dat_1=outlierr_c(np.copy(bg_dat_0),ks_b,sig_b)
                c2,bg_dat_2=outlierr_c(np.copy(bg_dat_1),ks_b,sig_b)
                if trip==True:
                    c3,bg_dat_3=outlierr_c(np.copy(bg_dat_2),ks_b,sig_b)
                    bg_dat=np.copy(bg_dat_3)
                    tr=c1+c2+c3
                else:
                    bg_dat=np.copy(bg_dat_2)
                    tr=c1+c2
 
                ### replace row_data with corrected background data ###
                row_data[ed_t:ed_l]=bg_dat[:ed_l-ed_t]
                row_data[xwidth-ed_u:xwidth-ed_t]=bg_dat[ed_l-ed_t:]
                    
                ### do background fitting  ###
                bg_params=np.polyfit(bg_pix,bg_dat,o_b)
                bg_fullXf=(np.poly1d(bg_params))(xpix_ar)
                
                row_data-=bg_fullXf
                
                ### save background ###
                bkgd_params[o,t,j+y_start,:]=bg_params
                bkgd_sv[t,j+y_start,:]=bg_fullXf
                corr_sv[t,j+y_start,:]=row_data
                
                diff=np.nanmax(bg_dat_0-bg_dat_1)
                
                if ver==True:
                    #if tr>=1:
                    if (100.*diff/bg_fullXf[10])>10.:
                        plt.figure(201,figsize=(14,4))
                        plt.title('BAKGROUND FILTERING: OBJ='+str(int(o))+
                                  ' TIME='+str(int(t))+' ROW='+str(np.round(100.*float(j)/float(n_rows),5))+'%')
                        plt.plot(bg_pix,bg_dat_0,color='black',linewidth=6.0,zorder=1)
                        plt.plot(bg_pix,bg_dat_1,color='purple', linewidth=3.0,zorder=2)
                        plt.plot(bg_pix,bg_dat_2,color='blue',linewidth=3.0,zorder=3)
                        plt.figtext(0.15,0.7,c1,color='purple',fontsize=25)
                        plt.figtext(0.15,0.5,c2,color='blue',fontsize=25)
                        if trip==True:
                            plt.plot(bg_pix,bg_dat_3,color='green',linewidth=3.0,zorder=4)
                            plt.figtext(0.15,0.3,c3,color='green',fontsize=25)
                        plt.plot(xpix_ar,bg_fullXf,linewidth=1.0,color='red',zorder=5)
                        plt.show(block=False)
                        plt.close()
                    
            ################# VERBOSE OUTPUT #################
            if ver_full==True:
                plt.figure(103,figsize=(14,4))
                plt.title('FIXED: OBJ='+str(int(o))+' TIME='+str(int(t)))
                for j in range(0,n_rows):
                    plt.plot(xpix_ar,corr_sv[t,j+y_start,:],linewidth=1.0)
                plt.axvline(x=ed_l, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
                plt.axvline(x=ed_t, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_t,color='grey',linewidth=0.5)
                plt.show(block=False)
            ###################################################
            
            if ver_t==True:
                if t%10==0:
                    fig,ax=plt.subplots(2,1,figsize=(15,4))
                    fig.subplots_adjust(wspace=0, hspace=0)
                    ax[0].imshow((obj_data[t,:,:].T),cmap=plt.cm.plasma,aspect='auto')
                    ax[0].set_xticklabels([])
                    ax[1].imshow((corr_sv[t,:,:].T),cmap=plt.cm.plasma,aspect='auto')
                    ax[1].set_xlim(y_start,obj_data.shape[1]+y_start)
                    ax[1].set_xticklabels([])
                    plt.figtext(0.15,0.70,t,color='white',fontsize=30)
                    plt.show(block=False)
                    plt.close()
         
        print datetime.now()-time0
        PARAMS=[extray,binnx,binny,Lflat,Ldark,ed_l,ed_u,ed_t,ks_b,sig_b,o_b,
                ver,ver_full,ver_t,trip,time_start,time_trim]
        np.savez_compressed(SAVEPATH+'BG_SUBTRACTION_'+str(int(o))+'.npz',params=PARAMS,
                            bkgd=bkgd_sv,corr=corr_sv,bkgd_params=bkgd_params)
    return
                        
                      
                    
                    
