import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip

import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import matplotlib

from datetime import datetime

from setup import *

from mF_outlier_removal import outlierr_c
from mF_outlier_removal import outlierr_model


def Gaussian_P(x,a,b,c,curv,slop,inte):
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))+(curv**x*x)*(slop*x)+inte

def Aperture(SAVEPATH, extray, binnx, binny, ed_t, time_start,time_trim, obj_skip, CON, apsize_in,ks, ver_sum):

    #SAVEPATH= location of saved 2D spec
    
    #CON = use constant aperture
    #apsize = multiple of FWHM to use
    #      If CON== 0, calculates median aperture across all exposures, wavelengths *apsize
    #      If CON== 1, uses median aperture in a given exposures *apsize
    #      If CON== 2, smooths aperture as a function of wavelength in each expousre *apsize
    
    PARAMS=[extray, binnx, binny, time_start,time_trim, obj_skip, CON, apsize_in]
    
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    print(n_obj,n_exp)
    
    flat_spec = np.empty([n_obj,n_exp,int(2*ypixels/binny+ygap)])*np.nan
    nons_spec = np.empty([n_obj,n_exp,int(2*ypixels/binny+ygap)])*np.nan
    
    norm=matplotlib.colors.Normalize(vmin=0,vmax=n_exp)
    colors=matplotlib.cm.viridis
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])
    
    for i in range(0,n_obj):
        if i in obj_skip:
            continue
            
        if len(apsize_in)>1:
            apsize=apsize_in[i]
        else:
            apsize=apsize_in
            
            
        time0=datetime.now()
        print('-----------------')
        print('  OBJECT # ', i)
        print('-----------------')
        
        print(' ----> loading data...')
        Spec2D = np.load(SAVEPATH+'2DFit_Obj'+str(int(i))+'.npz')['data']  #[n_exp,int(2*ypixels/binny+ygap),int(xwidth)]
        BKGRND = np.load(SAVEPATH+'2DFit_Obj'+str(int(i))+'.npz')['background']  #[n_exp,int(2*ypixels/binny+ygap),int(xwidth)]
        FWHM_f = np.load(SAVEPATH+'2DFit_Obj'+str(int(i))+'.npz')['gaus'][:,:,2]  
               #[n_exp,int(2*ypixels/binny+ygap),gaussian params]
               #third dimension: 1 = raw x-center fit
               #  2 = stndrd deviation w/ fwhm = 2*np.sqrt(2*np.ln(2)*c) (fwhm array in savedata already converted)
               
        X0_fit = np.load(SAVEPATH+'2DFit_Obj'+str(int(i))+'.npz')['x_fit']
        X0_fit -= 1.0  #convert pixel value to index value (start at 1 -> start at 0)
        print(X0_fit.shape)
        
        xwidth = np.int(Spec2D.shape[2])
        print('      (done)')
        
        
        if CON == 0:
            medfwhm = 2.*np.sqrt(2.*np.log(2.))*np. nanmedian(FWHM_f)
            xmin = np.zeros([n_exp,int(2*ypixels/binny+ygap)],dtype=int)
            xmax = np.zeros([n_exp,int(2*ypixels/binny+ygap)],dtype=int)
            
            for t in range(time_start,n_exp-time_trim):

                xmin[t,:] = np.floor(X0_fit[t,:] - apsize*medfwhm).astype(int)
                xmax[t,:] = np.ceil(X0_fit[t,:] + apsize*medfwhm).astype(int)
               

                for j in range(0, int(2*ypixels/binny+ygap)):
                    x_low = np.nanmax([0, xmin[t,j]])
                    x_top = np.nanmin([xwidth, xmax[t,j]])
                    flat_spec[i,t,j] = np.nansum(Spec2D[t,j,x_low:x_top+1])
                    nons_spec[i,t,j] = np.nansum(Spec2D[t,j,ed_t:x_low])+np.nansum(Spec2D[t,j,x_top+1:xwidth-ed_t])
                    
                if t==0:

                    print('    -> MEDIAN FWHM: ', np.round(medfwhm,2), np.int(apsize*medfwhm))
                    ###########
                    plt.figure(104,figsize=(10,4))
                    pltymin=np.nanmin(xmin[t,:]-medfwhm)
                    pltymax=np.nanmax(xmax[t,:]+medfwhm)

                    vmin = np.nanmin((Spec2D)[0,:,np.int(pltymin):np.int(pltymax)])+np.nanmean(BKGRND)
                    vmax = np.nanmax((Spec2D)[0,:,np.int(pltymin):np.int(pltymax)])+np.nanmean(BKGRND)

                    plt.imshow(((Spec2D)[0,:,:].T)+np.nanmean(BKGRND),cmap=plt.cm.plasma,aspect='auto',origin='lower',
                               extent=(0, 2*ypixels/binny+ygap, 0, xwidth),
                               norm=mpcolors.LogNorm(vmin,vmax))
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),xmin[t,:],color='white',linewidth=2.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),X0_fit[t,:],color='white',linewidth=4.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),xmax[t,:],color='white',linewidth=2.0,zorder=4)
                    plt.ylim(pltymin,pltymax)
                    plt.xlim(0,2*ypixels/binny+ygap)
                    plt.show(block=False)
                    plt.close()

                    if ver_sum == True:
                        if j%100 == True:
                            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,4))
                            ax.set_facecolor('darkgrey')
                            for t in range(0,n_exp):
                                plt.plot(np.linspace(1,xwidth,xwidth),
                                         Spec2D[t,j,:],color='black',lw=3.0,alpha=0.1)
                            for t in range(0,n_exp):
                                plt.plot(np.linspace(1,xwidth,xwidth)[x_low:x_top],
                                         Spec2D[t,j,x_low:x_top],color='tomato',lw=1.0,alpha=0.1)
                            plt.axvline(x=ed_t,color='black',lw=2.0, ls= '--')
                            plt.axvline(x=xwidth-ed_t,color='black',lw=2.0, ls= '--')
                            plt.axvline(x=x_low,color='red',lw=2.5, ls='-')
                            plt.axvline(x=x_top,color='red',lw=2.5, ls='-')
                            plt.axhline(y=0.0,color='blue',lw=2.5,ls='-')
                            plt.xlim(1,xwidth)
                            plt.ylim(-100,20)
                            plt.show(block=False)
                            plt.close()
                
        if CON == 1:
            medfwhm = 2.*np.sqrt(2.*np.log(2.))*np.nanmedian(FWHM_f,axis =1)
            print(medfwhm.shape)
            xmin = np.zeros([n_exp,int(2*ypixels/binny+ygap)],dtype=int)
            xmax = np.zeros([n_exp,int(2*ypixels/binny+ygap)],dtype=int)
            for t in range(time_start,n_exp-time_trim):
                xmin[t,:] = np.floor(X0_fit[t,:] - apsize*medfwhm[t]).astype(int)
                xmax[t,:] = np.ceil(X0_fit[t,:] + apsize*medfwhm[t]).astype(int)
                
                if t==0:
                    ###########
                    plt.figure(104,figsize=(10,4))
                    
                    
                    pltymin=np.nanmin(xmin[t,:]-medfwhm[t])
                    pltymax=np.nanmax(xmax[t,:]+medfwhm[t])
                    
                    vmin = np.nanmin((Spec2D)[t,:,np.int(pltymin):np.int(pltymax)])+np.nanmean(BKGRND)
                    vmax = np.nanmax((Spec2D)[t,:,np.int(pltymin):np.int(pltymax)])+np.nanmean(BKGRND)
                    
                    plt.imshow(((Spec2D)[t,:,:].T)+np.nanmean(BKGRND),cmap=plt.cm.plasma,aspect='auto',origin='lower',
                               extent=(0, 2*ypixels/binny+ygap, 0, xwidth),
                               norm=mpcolors.LogNorm(vmin,vmax))
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),xmin[t,:],color='white',linewidth=2.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),X0_fit[t,:],color='white',linewidth=4.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),xmax[t,:],color='white',linewidth=2.0,zorder=4)

                    plt.ylim(pltymin,pltymax)
                    plt.xlim(0,2*ypixels/binny+ygap)
                    plt.show(block=False)
                    plt.close()

                for j in range(0, int(2*ypixels/binny+ygap)):
                    x_low = np.nanmax([0, xmin[t,j]])
                    x_top = np.nanmin([xwidth, xmax[t,j]])
                    flat_spec[i,t,j] = np.nansum(Spec2D[t,j,x_low:x_top+1])
                    nons_spec[i,t,j] = np.nansum(Spec2D[t,j,ed_t:x_low])+np.nansum(Spec2D[t,j,x_top+1:xwidth-ed_t])

            plt.figure(105, figsize=(10,4))
            plt.plot(np.linspace(0,n_exp,n_exp),apsize*2.*np.sqrt(2.*np.log(2.))*FWHM_f[:,0], color='grey', alpha=0.5,lw=2.0)
            plt.plot(np.linspace(0,n_exp,n_exp),apsize*medfwhm,color='red',lw=2.0)
            plt.show(block=False)
            plt.close()
                 
                    
        if CON == 2:
            medfwhm = 2.*np.sqrt(2.*np.log(2.))*FWHM_f
            print(medfwhm.shape)
            
            xmin = np.zeros([n_exp,int(2*ypixels/binny+ygap)],dtype=int)
            xmax = np.zeros([n_exp,int(2*ypixels/binny+ygap)],dtype=int)
            
            fwhm_smooth = np.zeros_like(medfwhm)*0.0
            
            for t in range(time_start,n_exp-time_trim):
                medfwhm_t = medfwhm[t,:]
                naninds = np.where(np.isnan(medfwhm_t))[0]
                y_arr_nnan=np.linspace(1,2*ypixels/binny+ygap,
                                   2*ypixels/binny+ygap)[~np.isnan(medfwhm_t)]
                
                fwhm_smooth[t,:] = np.copy(medfwhm_t)
                for k in ks:
                    fwhm_smooth[t,:]=medfilt(fwhm_smooth[t,:],kernel_size=k)
                
                
                fwhm_smooth[t,naninds] = np.nan
                
                xmin[t,:] = np.floor(X0_fit[t,:] - apsize*fwhm_smooth[t,:]).astype(int)
                xmax[t,:] = np.ceil(X0_fit[t,:] + apsize*fwhm_smooth[t,:]).astype(int)
                
                if t==0:
                    plt.figure(2,figsize=(10,4))
                    
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                       2*ypixels/binny+ygap), medfwhm_t, color='grey', lw=2.0,alpha=0.5)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                       2*ypixels/binny+ygap), fwhm_smooth[t,:], color='red', lw=2.0)
                    
                    plt.ylim(np.nanmin(fwhm_smooth[t,:])-0.1, np.nanmax(fwhm_smooth[t,:])+0.1)
                    
                    plt.xlim(0,2*ypixels/binny+ygap)
                    plt.show(block=False)
                    plt.close()
                    
                    #######
                    plt.figure(104,figsize=(10,4))

                    pltymin=np.nanmin(xmin[t,:]-fwhm_smooth[t,:])
                    pltymax=np.nanmax(xmax[t,:]+fwhm_smooth[t,:])

                    vmin = np.nanmin((Spec2D)[t,:,np.int(pltymin):np.int(pltymax)])+np.nanmean(BKGRND)
                    vmax = np.nanmax((Spec2D)[t,:,np.int(pltymin):np.int(pltymax)])+np.nanmean(BKGRND)

                    plt.imshow(((Spec2D)[t,:,:].T)+np.nanmean(BKGRND),cmap=plt.cm.plasma,aspect='auto',origin='lower',
                               extent=(0, 2*ypixels/binny+ygap, 0, xwidth),
                               norm=mpcolors.LogNorm(vmin,vmax))
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),xmin[t,:],color='white',linewidth=2.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),X0_fit[t,:],color='white',linewidth=4.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,
                                           2*ypixels/binny+ygap),xmax[t,:],color='white',linewidth=2.0,zorder=4)

                    plt.ylim(pltymin,pltymax)
                    plt.xlim(0,2*ypixels/binny+ygap)
                    plt.show(block=False)
                    plt.close()

                for j in range(0, int(2*ypixels/binny+ygap)):
                    x_low = np.nanmax([0, xmin[t,j]])
                    x_top = np.nanmin([xwidth, xmax[t,j]])
                    flat_spec[i,t,j] = np.nansum(Spec2D[t,j,x_low:x_top+1])
                    nons_spec[i,t,j] = np.nansum(Spec2D[t,j,ed_t:x_low])+np.nansum(Spec2D[t,j,x_top+1:xwidth-ed_t])
                    
            plt.figure(105, figsize=(10,4))
            for j in range(0,int(2*ypixels/binny+ygap)):
                if j%10==0:
                    plt.plot(np.linspace(0,n_exp,n_exp),apsize*fwhm_smooth[:,j],color='tomato',lw=1.0,alpha=0.2)
            plt.show(block=False)
            plt.close()


        plt.figure(1,figsize=(10,4))
        for t in range(time_start,n_exp-time_trim):
            plt.plot(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap),flat_spec[i,t,:], color=scal_m.to_rgba(t),lw=1.0)
        plt.show(block=False)
        plt.close()

#         fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,4))
#         ax.set_facecolor('darkgrey')
#         for t in range(time_start,n_exp-time_trim):
#             plt.plot(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap),
#                      np.nansum(Spec2D[t,:,ed_t:xwidth-ed_t],axis=1)-flat_spec[i,t,:], color=scal_m.to_rgba(t),lw=1.0)
#         plt.ylim(-15000,15000)
#         plt.show(block=False)
#         plt.close()
        
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,4))
        ax.set_facecolor('darkgrey')
        for t in range(time_start,n_exp-time_trim):
            plt.plot(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap),
                     nons_spec[i,t,:], color=scal_m.to_rgba(t),lw=1.0)
        plt.ylim(-15000,15000)
        plt.show(block=False)
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,4))
        ax.set_facecolor('darkgrey')
        for t in range(time_start,n_exp-time_trim):
            plt.plot(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap),
                     np.nansum(BKGRND[t,:,ed_t:xwidth-ed_t],axis=1), color=scal_m.to_rgba(t),lw=1.0)
        plt.show(block=False)
        plt.close()

    np.savez(SAVEPATH+'FlattenedSpectra_CON'+str(CON)+'_AP'+str(int(apsize_in[0]*100)).zfill(3)+'.npz',params = PARAMS, flat_spec = flat_spec, nons_spec = nons_spec)
            
            
        
        
        
        

    