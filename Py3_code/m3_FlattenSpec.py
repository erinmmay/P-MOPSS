import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib

from datetime import datetime

from setup import *

from mF_outlier_removal import outlierr_c
from mF_outlier_removal import outlierr_model

def Gaussian_P(x,a,b,c,curv,slop,inte):
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))+(curv**x*x)*(slop*x)+inte


def FlattenSpec(extray,SAVEPATH,ed_l,ed_u,ed_t,binnx,binny,fb,CON,LVAR,
                ks_b,sig_b,ks_d,sig_d,ks_s,sig_s,ing_fwhm,ver_full,ver_fit,ver_spec,ver_xcen,
                data_corr,nruns,time_start,time_trim,obj_skip,reloadd,saver,r_skip,a_s,a_d):
    #extray= number of pixels in y direction extra that were extracted
    #SAVEPATH= location of saved 2D spec
    #ed_l= location of lower boundary between background/data - only used for initial guess of centroid
    #ed_u= location of upper boundary between background/data - only used for initial guess of centroid
    #ed_t= number of pixels on edges of 2D strip to trim - only used for initial guess of centroid
    #binnx= binning in x direction
    #binny= binning in y direction
    #Lflat, Ldark = Boolean to use Flats/Darks
    #CON = constant aperture
    #ks_b= kernel size for background outlier detection
    #sig_b= sigma threshold for background outliers
    #ing_fwmh - initial guess for fwhm
    #ver_* - various outputs, Boolean. See notebook
    #trip = Boolean. Run median filter 3 times
    #time_start/time_end = number of frames to trim on either side
    #obj_skip = don't run for these objects
    #a_s= aperture size to use
    #a_d= not used currently
    
    ###########################################
    PARAMS=[extray,ed_l,ed_u,ed_t,binnx,binny,Lflat,Ldark,CON,
                ks_b,sig_b,ks_d,sig_d,ks_s,sig_s,ing_fwhm,ver_full,ver_fit,ver_spec,
                data_corr,nruns,time_start,time_trim,a_s,a_d]
    
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    print(n_obj,n_exp)
    
    #bkgd_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,o_b+1])*np.nan
    
#     if Ldark==True:
#         dark_var=np.load(SAVEPATH+'Darks.npz')['var']
#     if Lflat==True:
#         flat_var=np.load(SAVEPATH+'Flats.npz')['var']
        
    fit_params=np.empty([n_obj,n_exp,int(2*ypixels/binny+ygap),6])*np.nan
    fwhm_data=np.empty([n_obj,n_exp])*np.nan
    flat_spec=np.empty([n_obj,n_exp,int(2*ypixels/binny+ygap)])*np.nan
    flat_bkgd=np.empty([n_obj,n_exp,int(2*ypixels/binny+ygap)])*np.nan
    
    for i in range(0,n_obj):
        if i in obj_skip:
            continue
            
        time0=datetime.now()
        print('-----------------')
        print('  OBJECT # ', i)
        print('-----------------')
        
        print(' ----> loading data...')
        obj_data=(np.load(SAVEPATH+'2DSpec_obj'+str(int(i))+'.npz'))['data']
        print('      (done)')
        
        print(' ----> loading masks...')
        mask=(np.load(SAVEPATH+'FinalMasks.npz')['masks'])[i,:]
        print('      (done)')
        
        y0=int(mask[1])  #pixel number of inital extraction
        #y_start=np.int(np.max([0,y0-extray]))  #including extray
        ### BINN THE SIZES OF THE MASKS in Y ###
        if binny>1 and fb==1:
            if y0<=ypixels:
                y0=y0/binny
            if y0>ypixels and y0<=ypixels+ygap:
                dy=y0-ypixels
                y0=(y0-ypixels-dy)/binny+ypixels/binny+dy
            if y0>ypixels+ygap:
                y0=(y0-ypixels-ygap)/binny+ypixels/binny+ygap
             
        y_start=np.int(np.max([0,y0-extray]))  #including extray
        
        ##################################
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        print(i, n_rows,xwidth,y_start)
        
        xpix_ar=np.linspace(1,xwidth,xwidth)
        #fwhm_av=np.empty([n_exp])*np.nan
        
        bkgd_sv=np.empty([n_exp,int(2*ypixels/binny+ygap),int(xwidth)])*np.nan
        #corr_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
    
        sub_bkgd=np.empty([n_exp,int(2*ypixels/binny+ygap),int(xwidth)])*np.nan
        
        
        #begin loop over time...
        for t in range(time_start,n_exp-time_trim):
            if t%10==0:
                print('       *** TIME: ',t,' ***')
            
            frame=np.copy(obj_data[t,:,:])   #current frame
            plt_data=np.empty([int(2*ypixels/binny+ygap),int(xwidth)])*np.nan
            
            ###################################################
            for j in range(0,n_rows):
                if j+y_start>=ypixels/binny and j+y_start<ypixels/binny+ygap:
                    continue
                row_data=np.copy(frame[j,:])  #obj data is not y_start shifted 
                #print ed_l,ed_u

                

             # flatten spec....  
            for j in range(0,n_rows):
                if LVAR==False:
                    lowi=int(np.nanmax([fit_params[i,t,j+y_start,1]-a_s*fwhm_data[i,t],0]))
                    uppi=int(np.nanmin([fit_params[i,t,j+y_start,1]+a_s*fwhm_data[i,t],xwidth]))
                else:
                    lowi=int(np.nanmax([fit_params[i,t,j+y_start,1]-a_s*fit_params[i,t,j+y_start,2],0]))
                    uppi=int(np.nanmin([fit_params[i,t,j+y_start,1]+a_s*fit_params[i,t,j+y_start,2],xwidth]))
                #lowi=int(np.nanmax([fit_params[i,t,j+y_start,1]-a_s*fit_params[i,t,j+y_start,2],0]))
                #uppi=int(np.nanmin([fit_params[i,t,j+y_start,1]+a_s*fit_params[i,t,j+y_start,2],xwidth]))
                flat_spec[i,t,j+y_start]=np.nansum(sub_bkgd[t,j+y_start,lowi:uppi])
                flat_bkgd[i,t,j+y_start]=np.nansum(bkgd_sv[t,j+y_start,lowi:uppi])
                
        rm_s=0
        for t in range(time_start,n_exp-time_trim):
            rm_s_a,flat_spec[i,t,:]=outlierr_c(np.copy(flat_spec[i,t,:]),ks_s,sig_s)
            rm_s+=rm_s_a
            rm_s_a,flat_spec[i,t,:]=outlierr_c(np.copy(flat_spec[i,t,:]),ks_s,sig_s)
            rm_s+=rm_s_a
            #if trip==True:
            #    rm_s_a,flat_spec[i,t,:]=outlierr_c(np.copy(flat_spec[i,t,:]),ks_s,sig_s)
            #    rm_s+=rm_s_a
        
        if ver_spec==True:
            norm=matplotlib.colors.Normalize(vmin=0,vmax=n_exp)
            colors=matplotlib.cm.viridis
            scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
            scal_m.set_array([])

            fig=plt.figure(301,figsize=(15,4))
            for t in range(time_start,n_exp-time_trim):
                plt.plot(flat_spec[i,t,:],color=scal_m.to_rgba(t),linewidth=1.0)
            # [left, bottom, width, height
            cbaxes = fig.add_axes([0.15, 0.2, 0.02, 0.6]) 
            cb = plt.colorbar(scal_m, cax = cbaxes)  
            plt.figtext(0.2,0.8,rm_s,color='black',fontsize=25)
            plt.show(block=False)
            plt.close()
        print(datetime.now()-time0) 
        np.savez_compressed(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz',params=PARAMS,spec=flat_spec[i,:,:],
                            bkgd=flat_bkgd[i,:,:], fwhm=fwhm_data[i,:], gaus=fit_params[i,:,:,:])
   
    if saver==True:
        if reloadd==True:
            print('---- reloading in previously run targets')
            for i in range(0,n_obj):
                if i in r_skip:
                    continue
                print('         ',i)
                flat_spec[i,:,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['spec']
                flat_bkgd[i,:,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['bkgd']
                fwhm_data[i,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['fwhm']
                fit_params[i,:,:,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['gaus']    
        print('---- saving data')
        np.savez_compressed(SAVEPATH+'FlattenedSpectra.npz',params=PARAMS,flat_spec=flat_spec,flat_bkgd=flat_bkgd,
                            fwhm_ar=fwhm_data,gaus_params=fit_params)