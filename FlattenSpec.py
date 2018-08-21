import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib

from datetime import datetime

from setup import *

from outlier_removal import outlierr_c
from outlier_removal import outlierr_model

def Gaussian_P(x,a,b,c,x0,x1):
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))+(x0*x)+x1


def FlattenSpec(extray,SAVEPATH,ed_l,ed_u,ed_t,binnx,binny,fb,Lflat,Ldark,CON,
                ks_b,sig_b,ks_d,sig_d,ks_s,sig_s,ing_fwhm,ver_full,ver_fit,ver_spec,ver_xcen,
                data_corr,trip,time_start,time_trim,obj_skip,reloadd,saver,r_skip,a_s,a_d):
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
                data_corr,trip,time_start,time_trim,a_s,a_d]
    
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    
    #bkgd_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,o_b+1])*np.nan
    
#     if Ldark==True:
#         dark_var=np.load(SAVEPATH+'Darks.npz')['var']
#     if Lflat==True:
#         flat_var=np.load(SAVEPATH+'Flats.npz')['var']
        
    fit_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,5])*np.nan
    fwhm_data=np.empty([n_obj,n_exp])
    flat_spec=np.empty([n_obj,n_exp,2*ypixels/binny+ygap])*np.nan
    flat_bkgd=np.empty([n_obj,n_exp,2*ypixels/binny+ygap])*np.nan
    
    for i in range(0,n_obj):
        if i in obj_skip:
            continue
            
        time0=datetime.now()
        print '-----------------'
        print '  OBJECT # ', i
        print '-----------------'
        
        print ' ----> loading data...'
        obj_data=(np.load(SAVEPATH+'2DSpec_obj'+str(int(i))+'.npz'))['data']
        print '      (done)'
        
        print ' ----> loading masks...'
        mask=(np.load(SAVEPATH+'FinalMasks.npz')['masks'])[i,:]
        print '      (done)'
        
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
        print i, n_rows,xwidth,y_start
        
        xpix_ar=np.linspace(1,xwidth,xwidth)
        #fwhm_av=np.empty([n_exp])*np.nan
        
        bkgd_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        #corr_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
    
        sub_bkgd=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        
        
        #begin loop over time...
        for t in range(time_start,n_exp-time_trim):
            if t%10==0:
                print '       *** TIME: ',t,' ***'
            
            frame=np.copy(obj_data[t,:,:])   #current frame
            plt_data=np.empty([2*ypixels/binny+ygap,xwidth])*np.nan
            
            ################# VERBOSE OUTPUT #################
            if ver_full==True:
                if t%10==0:
                    plt.figure(102,figsize=(14,4))
                    plt.title('RAW: OBJ='+str(int(i))+' TIME='+str(int(t)))
                    for j in range(0,n_rows):
                        plt.plot(xpix_ar,frame[j,:],linewidth=1.0)
                    plt.axvline(x=ed_l, color='grey',linewidth=0.5)
                    plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
                    plt.axvline(x=ed_t, color='grey',linewidth=0.5)
                    plt.axvline(x=xwidth-ed_t,color='grey',linewidth=0.5)
                    plt.show(block=False)
                    plt.close()
                else:
                    print '*** ', t, ' ***'
            ###################################################
            for j in range(0,n_rows):
                if j+y_start>=ypixels/binny and j+y_start<ypixels/binny+ygap:
                    continue
                row_data=np.copy(frame[j,:])  #obj data is not y_start shifted 
                
                ## FIT IN COMBINATION -- polynomial and gaussian ##
                p0=np.array([(np.nanmax(row_data[ed_l:xwidth-ed_u])-np.nanmedian(row_data[ed_t:ed_l])),
                             xpix_ar[np.argmax(row_data[ed_l:xwidth-ed_u])]+ed_l,
                             ing_fwhm,0.0,
                             np.nanmedian(row_data[ed_t:ed_l])])
                #check S/N of row:
                if np.nanmax(row_data[ed_l:xwidth-ed_u])<1.5*np.nanmedian(row_data[ed_t:ed_l]):
                    frame[j,:]=np.empty([len(row_data)])*np.nan
                    if t%10==0:
                        print '        LOW S/N AT ROW', j+y_start
                    continue
                        
                #if t%10==0 and j%100==0:
                #    print t, j, p0
                try:
                    g_param,g_cov=curve_fit(Gaussian_P,xpix_ar,row_data,p0=p0,maxfev=100000)
                except RuntimeError:
                    fit_params[i,t,j+y_start,:]=fit_params[i,t,j+y_start-1,:]
                else:
                    fit_params[i,t,j+y_start,:]=g_param
                    
                #### fitting plots -ver_fit ###
                if ver_fit==True:
                    if t%10==0:
                        fig,ax=plt.subplots(2,1,figsize=(15,6))
                        fig.subplots_adjust(wspace=0, hspace=0)
                        ax[0].plot(xpix_ar,row_data,color='black',linewidth=4.0)

                        ax[0].plot(xpix_ar,Gaussian_P(xpix_ar,*p0),
                                   color='blue',linewidth=1.0,linestyle='--')
                        ax[0].plot(xpix_ar,Gaussian_P(xpix_ar,*fit_params[i,t,j+y_start,:]),
                                   color='cyan',linewidth=2.0)
                        ax[0].axvline(x=fit_params[i,t,j+y_start,1],color='red',linewidth=2.0)
                        ax[0].axvline(x=fit_params[i,t,j+y_start,1]-a_s*fit_params[i,t,j+y_start,2],
                                      color='tomato',linewidth=1.0)
                        ax[0].axvline(x=fit_params[i,t,j+y_start,1]+a_s*fit_params[i,t,j+y_start,2],
                                      color='tomato',linewidth=1.0) 
                        ax[0].set_xlim(0,xwidth)
                        plt.figtext(0.2,0.75,str(int(i))+' '+str(int(t))+' '+str(int(j)),fontsize=20)
  

                ### run median filter compared to first fit ON BACKGROUND ONLY###
                ned_l=int(np.nanmax([fit_params[i,t,j+y_start,1]-
                                     (a_s)*fit_params[i,t,j+y_start,2],0]))
                ned_u=int(np.nanmin([fit_params[i,t,j+y_start,1]+
                                     (a_s)*fit_params[i,t,j+y_start,2],xwidth]))
                if ned_l>xwidth:
                    ned_l=0
                if ned_u<0:
                    ned_u=xwidth
                #print t,j, ned_l,ned_u
        
                nxpix_ar=np.append(xpix_ar[:ned_l],xpix_ar[ned_u:])
                nrow_data=np.append(row_data[:ned_l],row_data[ned_u:])
                
                model=Gaussian_P(nxpix_ar,*fit_params[i,t,j+y_start,:])
#                 plt.figure(201,figsize=(10,10))
#                 plt.plot(xpix_ar,row_data,color='black',linewidth=5.0)
#                 plt.plot(nxpix_ar,model,color='red',linewidth=2.0)
#                 plt.show()
                c1=0
                c2=0
                c3=0
                c1,row_data_1=outlierr_c(np.copy(nrow_data),ks_b,sig_b)
                c2,row_data_2=outlierr_c(np.copy(row_data_1),ks_b,sig_b)
                if trip==True:
                    c3,row_data_3=outlierr_c(np.copy(row_data_2),ks_b,sig_b)
                    row_data[:ned_l]=np.copy(row_data_3[:ned_l])
                    row_data[ned_u:]=np.copy(row_data_3[ned_l:])
                    tr=c1+c2+c3
                else:
                    row_data[:ned_l]=np.copy(row_data_2[:ned_l])
                    row_data[ned_u:]=np.copy(row_data_2[ned_l:])
                    tr=c1+c2
                ### refit- model ###
                ## FIT IN COMBINATION -- polynomial and gaussian ##
                p0=np.array([(np.nanmax(row_data[ed_l:xwidth-ed_u])-np.nanmedian(row_data[ed_t:ed_l])),
                             xpix_ar[np.argmax(row_data[ed_l:xwidth-ed_u])]+ed_l,
                             ing_fwhm,0.0,
                             np.nanmedian(row_data[ed_t:ed_l])])
                try:
                    g_param,g_cov=curve_fit(Gaussian_P,xpix_ar,row_data,p0=p0,maxfev=100000)
                except RuntimeError:
                    fit_params[i,t,j+y_start,:]=fit_params[i,t,j+y_start-1,:]
                else:
                    fit_params[i,t,j+y_start,:]=g_param    
                
                if data_corr==True:
                    ### run median filter compared to first fit ON BACKGROUND ONLY###
                    ned_l=int(np.nanmax([fit_params[i,t,j+y_start,1]-
                                         (a_s)*fit_params[i,t,j+y_start,2],0]))
                    ned_u=int(np.nanmin([fit_params[i,t,j+y_start,1]+
                                         (a_s)*fit_params[i,t,j+y_start,2],xwidth]))
                    if ned_l>xwidth:
                        ned_l=0
                    if ned_u<0:
                        ned_u=xwidth
                    #print t,j, ned_l,ned_u

                    nxpix_ar=np.copy(xpix_ar[ned_l:ned_u])
                    nrow_data=np.copy(row_data[ned_l:ned_u])

                    model=Gaussian_P(nxpix_ar,*fit_params[i,t,j+y_start,:])
    #                 plt.figure(201,figsize=(10,10))
    #                 plt.plot(xpix_ar,row_data,color='black',linewidth=5.0)
    #                 plt.plot(nxpix_ar,model,color='red',linewidth=2.0)
    #                 plt.show()
                    c1=0
                    c2=0
                    c3=0
                    c1,row_data_1=outlierr_model(np.copy(nrow_data),model,ks_d,sig_d)
                    c2,row_data_2=outlierr_model(np.copy(row_data_1),model,ks_d,sig_d)
                    if trip==True:
                        c3,row_data_3=outlierr_model(np.copy(row_data_2),model,ks_d,sig_d)
                        row_data[ned_l:ned_u]=np.copy(row_data_3)
                        n=c1+c2+c3
                        tr+=n
                    else:
                        row_data[ned_l:ned_u]=np.copy(row_data_2)
                        n=(c1+c2)
                        tr+=n
                        
                    ### refit- model ###
                    ## FIT IN COMBINATION -- polynomial and gaussian ##
                    p0=np.array([(np.nanmax(row_data[ed_l:xwidth-ed_u])-
                                  np.nanmedian(row_data[ed_t:ed_l])),
                                 xpix_ar[np.argmax(row_data[ed_l:xwidth-ed_u])]+ed_l,
                                 ing_fwhm,0.0,
                                 np.nanmedian(row_data[ed_t:ed_l])])
                    try:
                        g_param,g_cov=curve_fit(Gaussian_P,xpix_ar,row_data,p0=p0,maxfev=100000)
                    except RuntimeError:
                        fit_params[i,t,j+y_start,:]=fit_params[i,t,j+y_start-1,:]
                    else:
                        fit_params[i,t,j+y_start,:]=g_param 
                    
                
                if ver_fit==True:
                    if t%10==0:
                        ax[1].plot(nxpix_ar,row_data_1,color='grey',linewidth=2.0)
                        if trip==True:
                            ax[1].plot(nxpix_ar,row_data_2,color='grey',linewidth=2.0)
                        ax[1].plot(xpix_ar,row_data,color='black',linewidth=4.0)
                        ax[1].plot(xpix_ar,Gaussian_P(xpix_ar,*fit_params[i,t,j+y_start,:]),
                                   color='cyan',linewidth=2.0)
                        ax[1].axvline(x=fit_params[i,t,j+y_start,1],color='red',linewidth=2.0)
                        ax[1].axvline(x=fit_params[i,t,j+y_start,1]-a_s*fit_params[i,t,j+y_start,2],
                                      color='tomato',linewidth=1.0)
                        ax[1].axvline(x=fit_params[i,t,j+y_start,1]+a_s*fit_params[i,t,j+y_start,2],
                                      color='tomato',linewidth=1.0) 
                        ax[1].set_xlim(0,xwidth)
                        plt.figtext(0.2,0.4,tr,fontsize=20)                          
                        plt.show(block=False)
                        plt.close()
                        
                
                        
                bkgd_sv[t,j+y_start,:]=fit_params[i,t,j+y_start,3]*xpix_ar+fit_params[i,t,j+y_start,4]
                sub_bkgd[t,j+y_start,:]=row_data-bkgd_sv[t,j+y_start,:]
                frame[j,:]=row_data
                plt_data[j+y_start,:]=row_data-bkgd_sv[t,j+y_start,:]
                
            #calculating fit to x-centers
            y_arr_nnan=np.linspace(1,2*ypixels/binny+ygap,
                                   2*ypixels/binny+ygap)[~np.isnan(fit_params[i,t,:,1])]
            x_ctr_nnan=fit_params[i,t,~np.isnan(fit_params[i,t,:,1]),1]

            x_fit=np.polyfit(y_arr_nnan,medfilt(x_ctr_nnan,kernel_size=25),2)

            x_fit_nnan=(np.poly1d(x_fit))(y_arr_nnan)
            x_fit_full=(np.poly1d(x_fit))(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap))

            fit_params[i,t,:,1]=x_fit_full
            
            fwhm_data[i,t]=2*np.sqrt(2.*np.log(2.))*np.nanmedian(fit_params[i,t,:,2])
            
            if ver_xcen==True:
                if t%10==0:
                    plt.figure(104,figsize=(15,2))
                    plt.imshow((plt_data.T),cmap=plt.cm.plasma,aspect='auto')
                    plt.plot(y_arr_nnan,x_fit_nnan,color='white',linewidth=2.0,zorder=4)
                    plt.plot(y_arr_nnan,x_fit_nnan-a_s*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.plot(y_arr_nnan,x_fit_nnan+a_s*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.figtext(0.2,0.7,str(int(t)),fontsize=20)
                    plt.ylim(np.nanmin(x_fit_nnan-(a_s+1)*fwhm_data[i,t]),
                             np.nanmax(x_fit_nnan+(a_s+1)*fwhm_data[i,t]))
                    plt.show(block=False)
            
            ################# VERBOSE OUTPUT #################
            if ver_full==True:
                if t%10==0:
                    plt.figure(102,figsize=(14,4))
                    plt.title('CORRECTED: OBJ='+str(int(i))+' TIME='+str(int(t)))
                    for j in range(0,n_rows):
                        plt.plot(xpix_ar,sub_bkgd[t,j+y_start,:],linewidth=1.0)
                    plt.show(block=False)
                    plt.close()
            ###################################################
             # flatten spec....  
            for j in range(0,n_rows):
                lowi=int(np.nanmax([fit_params[i,t,j+y_start,1]-a_s*fwhm_data[i,t],0]))
                uppi=int(np.nanmin([fit_params[i,t,j+y_start,1]+a_s*fwhm_data[i,t],xwidth]))
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
            if trip==True:
                rm_s_a,flat_spec[i,t,:]=outlierr_c(np.copy(flat_spec[i,t,:]),ks_s,sig_s)
                rm_s+=rm_s_a
        
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
        print datetime.now()-time0  
        np.savez_compressed(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz',params=PARAMS,spec=flat_spec[i,:,:],
                            bkgd=flat_bkgd[i,:,:], fwhm=fwhm_data[i,t], gaus=fit_params[i,:,:,:])
   
    if saver==True:
        if reloadd==True:
            print '---- reloading in previously run targets'
            for i in range(0,n_obj):
                if i in r_skip:
                    continue
                print '         ',i
                flat_spec[i,:,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['spec']
                flat_bkgd[i,:,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['bkgd']
                fwhm_data[i,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['fwhm']
                fit_params[i,:,:,:]=np.load(SAVEPATH+'FlatSpec_Obj'+str(int(i))+'.npz')['gaus']    
        print '---- saving data'
        np.savez_compressed(SAVEPATH+'FlattenedSpectra.npz',params=PARAMS,flat_spec=flat_spec,flat_bkgd=flat_bkgd,
                            fwhm_ar=fwhm_data,gaus_params=fit_params)
               