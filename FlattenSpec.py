import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from datetime import datetime

from setup import *

def Gaussian(x,a,b,c,d):
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))+d

def Fit_Gaussian(i,t,j,y_start,x,y,p0,gaus_params):
    p0=p0
    try:
        g_param,g_cov=curve_fit(Gaussian,x,y,p0=p0,maxfev=10000)
    except RuntimeError:
#        sub_bkgd[t,j,:]=y-background
        gaus_params[i,t,j+y_start,:]=gaus_params[i,t,j+y_start-1,:]
    else:
#        sub_bkgd[t,j,:]=y-background
        gaus_params[i,t,j+y_start,:]=g_param
    return gaus_params
        

def median_filter(pixels,data,ks,std):
    data_median=medfilt(data,kernel_size=ks)
    
    data_median_r=data-data_median
    data_std=np.nanstd(data_median_r)
    data_med=np.nanmedian(data_median_r)
    
    counter=0
    for p in range(0,len(pixels)):
        p=int(p)
        if data_median_r[p]>std*data_std+data_med or data_median_r[p]<data_med-std*data_std:
            counter+=1
            data[p]=data_median[p]
            
    return counter,data

def gaussian_filter(ver,i,t,j,y_start,pixels,data,ks,std,fpixs,fdat,gaus_params):
    data_median=medfilt(data,kernel_size=ks)
    
    g_param=gaus_params[i,t,j+y_start,:]
    gfit=Gaussian(pixels,*g_param)
    
    data_median_r=data-gfit
    data_std=np.nanstd(data_median_r)
    data_med=np.nanmedian(data_median_r)
    
    counter=0
    
    
    for p in range(0,len(pixels)):
        p=int(p)
        if data_median_r[p]>std*data_std+data_med or data_median_r[p]<data_med-std*data_std:
            ################# VERBOSE OUTPUT #################
            if ver==True:
                fig,ax=plt.subplots(1,3,figsize=(9.,2.))
                            
                ax[0].plot(pixels,data,color='black',linewidth=2.0)
                ax[0].plot(pixels,gfit,color='cyan',linewidth=1.0)
                ax[0].axvline(x=g_param[1],color='grey')
                #plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                ax[0].set_title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                
                ax[1].plot(pixels,data_median_r,color='black',linewidth=2.0)

            ###################################################
            counter+=1
            index=(np.where(fpixs==pixels[p])[0][0])
            
            data[p]=data_median[p]
            fdat[index]=data_median[p]
            
            p0=g_param
            gaus_params=Fit_Gaussian(i,t,j,y_start,fpixs,fdat,p0,gaus_params)
            gfit=Gaussian(pixels,*gaus_params[i,t,j+y_start,:])
            ################# VERBOSE OUTPUT #################
            if ver==True:
                ax[1].plot(pixels,data-gfit,color='red',linewidth=1.0)
                ax[1].axhline(y=data_med,color='lime')
                ax[1].axhline(y=data_med+std*data_std,color='lime',linestyle='--')
                ax[1].axhline(y=data_med-std*data_std,color='lime',linestyle='--')
                ax[1].axhline(y=data_med+1.5*std*data_std,color='green',linestyle='--')
                ax[1].axhline(y=data_med-1.5*std*data_std,color='green',linestyle='--')
                #plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                ax[1].set_title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                
                ax[2].plot(pixels,data,color='black',linewidth=2.0)
                ax[2].plot(pixels,gfit,color='cyan',linewidth=1.0)
                ax[2].axvline(x=gaus_params[i,t,j+y_start,1],color='grey')
                #plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                ax[2].set_title(str(int(i))+' '+str(int(t))+' '+str(int(j+y_start)))
                        
                plt.show(block=False)
            ###################################################
                    
    return counter,fdat,gaus_params

def FlattenSpec(extray,SAVEPATH,corr,binnx,binny,Lflat,Ldark,CON,ed_l,ed_u,ed_t,ks_b,ks_d,sig_b,sig_d,ing_fwhm,
                ver,data_corr,trip,time_start,time_trim,obj_skip):
    #extray= number of pixels in y direction extra that were extracted
    #SAVEPATH= location of saved 2D spec
    #filename= name of saved file
    #savefile= name to save as
    #corr= cosmic ray corecction?
    #ed_l= location of lower boundary between background/data
    #ed_u= location of upper boundary between background/data
    #ed_t= number of pixels on edges of 2D strip to trim
    #ks_b= kernel size for background outlier detection
    #ks_d= kernel size for data area outlier detection
    #sig_b= sigma threshold for background outliers
    #sig_d= sigma thershold for data outliers
    #ver = verbose output (LOTS OF PLOTS!)
    #data_cor= run correction on gaussian?
    
    ###########################################
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    
    flat_spec=np.empty([n_obj,n_exp,2*ypixels/binny+ygap])*np.nan  #to store flattened data
    
    pht_err=np.zeros_like(flat_spec)                         #calculated before background subtraction!
    tot_err=np.zeros_like(flat_spec)                         #includes errors introduced by background subtraction
    
    gaus_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,4])*np.nan
    bkgd_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,2])*np.nan
    
    if Ldark==True:
        dark_var=np.load(SAVEPATH+'Darks.npz')['var']
    if Lflat==True:
        flat_var=np.load(SAVEPATH+'Flats.npz')['var']
    
    fwhm_av=np.empty([n_obj,n_exp])*np.nan
    
    #begin loop over objects...
    for i in range(0,n_obj):
        if i in obj_skip:
            continue
        time0=datetime.now()
        print '-----------------'
        print '  OBJECT # ', i
        print '-----------------'
        obj_data=(np.load(SAVEPATH+'2DSpec_obj'+str(int(i))+'.npz'))['data']
        
        
        #corrected=np.zeors_like(obj_data)*np.nan
        #sub_bkgd=np.zeros_like(obj_data)*np.nan #holds background subtracted data
        
        mask=(np.load(SAVEPATH+'FinalMasks.npz')['masks'])[i,:]
        y0=int(mask[1])  #pixel number of inital extraction
        ### BINN THE SIZES OF THE MASKS in Y ###
        if y0<ypixels:
            y0=y0/binny
        if y0>ypixels:
            y0=(y0-ypixels-ygap)/binny+ypixels/binny+ygap
        ##################################
        y_start=np.int(np.max([0,y0-extray]))  #including extray
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        
        xpix_ar=np.linspace(1,xwidth,xwidth)
        #fwhm_av=np.empty([n_exp])*np.nan
        
        bckgnd_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        corr_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
    
        sub_bkgd=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        
        #begin loop over time...
        for t in range(time_start,n_exp-time_trim):
            if t%10==0:
                print '    -->> TIME: ',t
                print '       -- Correcting and Fitting...'
            
            frame=np.array(obj_data[t,:,:])   #current frame
            ################# VERBOSE OUTPUT #################
            if ver==True:
                plt.figure(102,figsize=(14,4))
                plt.title('RAW: OBJ='+str(int(i))+' TIME='+str(int(t)))
                for j in range(0,n_rows):
                    plt.plot(xpix_ar,frame[j,:],linewidth=1.0)
                plt.axvline(x=ed_l, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
                plt.axvline(x=ed_t, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_t,color='grey',linewidth=0.5)
                plt.show(block=False)
            ###################################################
            
            #begin loop over row...
            for j in range(0,n_rows):
                row_data=np.array(frame[j,:])
                if not np.isfinite(row_data[0]):
                    continue
                    
                bg_pix=np.append(xpix_ar[ed_t:ed_l],xpix_ar[xwidth-ed_u:xwidth-ed_t])
                bg_dat=np.array(np.append(row_data[ed_t:ed_l],row_data[xwidth-ed_u:xwidth-ed_t]))
                
                #do background outlier detection and replacement.... (twice)
                n_replace1,bg_dat=median_filter(bg_pix,bg_dat,ks_b,sig_b)
                n_replace2,bg_dat=median_filter(bg_pix,bg_dat,ks_b,sig_b)
                if trip==True:
                    n_replace3,bg_dat=median_filter(bg_pix,bg_dat,ks_b,sig_b)
                
                #if t%10==0 and j%100==0:
                #    print '       -- ROW: ', j, '  removed ', n_replace1,n_replace2,' bg outliers'
                
                #fit linear function to background...
                bg_params=np.polyfit(bg_pix,bg_dat,1)
                background=(np.poly1d(bg_params))(xpix_ar)
                
                
                row_data[ed_t:ed_l]=bg_dat[:ed_l-ed_t]
                row_data[xwidth-ed_u:xwidth-ed_t]=bg_dat[ed_l-ed_t:]
                
                bckgnd_sv[t,j+y_start,:]=background
                bkgd_params[i,t,j+y_start,:]=bg_params
                corr_sv[t,j+y_start,:]=row_data
                
                if CON==False:
                #    print '       -- Fitting GAUSSIAN...'
                    #FIT GAUSSIAN WITHIN DATA REGION....
                    p0=np.array([np.nanmax(row_data)-background[20],np.argmax(row_data),ing_fwhm,background[20]])

                    gaus_params=Fit_Gaussian(i,t,j,y_start,xpix_ar,row_data,p0,gaus_params)

    #                 if ver==True:
    #                     if t%10==0:
    #                         if j%10==0:
    #                             print p0
    #                             print gaus_params[i,t,j+y_start,:]

                    if gaus_params[i,t,j+y_start,2]>20. and t>0:
                        gaus_params[i,t,j+y_start,:]=gaus_params[i,t-1,j+y_start,:]
                    if gaus_params[i,t,j+y_start,2]>20. and t==0:
                        gaus_params[i,t,j+y_start,:]=p0
#                     if ver==True:
#                         if t%10==0:
#                             if j%10==0:
#                                 plt.figure(201,figsize=(14,4))
#                                 plt.title('ROW='+str(int(j)))
#                                 plt.plot(xpix_ar,row_data,color='black',linewidth=2.0)
#                                 plt.plot(xpix_ar,Gaussian(xpix_ar,*p0),color='blue',linewidth=0.5)
#                                 plt.plot(xpix_ar,Gaussian(xpix_ar,*gaus_params[i,t,j+y_start,:]),color='cyan',linewidth=1.0)
#                                 plt.axvline(x=gaus_params[i,t,j+y_start,1],color='grey')
#                                 plt.show(block=False)

                    # corrections along gaussian...
                    if data_corr==True:

                        #if t%10==0:
                        #    print '              (Correcting GAUSSIAN)'
                        d_pix=xpix_ar[ed_l:xwidth-ed_u]
                        d_dat=row_data[ed_l:xwidth-ed_u]

                        if ver==True:
                            print '------------------------------------------------------------------'
                        dreplace1,row_data,gaus_params=gaussian_filter(
                             ver,i,t,j,y_start,d_pix,d_dat,ks_d,sig_d,xpix_ar,row_data,gaus_params)

                        dreplace2,row_data,gaus_params=gaussian_filter(
                             ver,i,t,j,y_start,d_pix,d_dat,ks_d,sig_d,xpix_ar,row_data,gaus_params)

                        if trip==True:
                            dreplace3,row_data,gaus_params=gaussian_filter(
                             ver,i,t,j,y_start,d_pix,d_dat,ks_d,sig_d,xpix_ar,row_data,gaus_params)

                    corr_sv[t,j+y_start,:]=row_data
                    
            if CON==False:   
                fwhm_av[i,t]=2.*np.sqrt(2.*np.log(2.))*np.nanmedian(gaus_params[i,t,:,2])
            if CON==True:
                fwhm_av[i,t]=ing_fwhm
                gaus_params[i,t,j+y_start,0]=np.nanmax(row_data)
                gaus_params[i,t,j+y_start,1]=np.argmax(row_data)
                gaus_params[i,t,j+y_start,2]=ing_fwhm
                gaus_params[i,t,j+y_start,3]=background[20]
            ################# VERBOSE OUTPUT #################
            if ver==True:
                plt.figure(103,figsize=(14,4))
                plt.title('FIXED: OBJ='+str(int(i))+' TIME='+str(int(t)))
                for j in range(0,n_rows):
                    plt.plot(xpix_ar,corr_sv[t,j+y_start,:],linewidth=1.0)
                plt.axvline(x=ed_l, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
                plt.axvline(x=ed_t, color='grey',linewidth=0.5)
                plt.axvline(x=xwidth-ed_t,color='grey',linewidth=0.5)
                plt.show(block=False)
            ###################################################
            if CON==False:
                if t%10==0:
                    print '       -- Fitting Centroid Function...'


                # calculting a fit for x-centers
                y_arr_nnan=np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap)[~np.isnan(gaus_params[i,t,:,1])]
                x_ctr_nnan=gaus_params[i,t,~np.isnan(gaus_params[i,t,:,1]),1]

                x_fit=np.polyfit(y_arr_nnan,x_ctr_nnan,3)

                x_fit_nnan=(np.poly1d(x_fit))(y_arr_nnan)
                x_fit_full=(np.poly1d(x_fit))(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap))

                #gaus_params[i,t,:,1]=x_fit_full

                ################# VERBOSE OUTPUT #################
                if ver==True:
                    plt.figure(104,figsize=(6,3))
                    plt.plot(y_arr_nnan,x_ctr_nnan,'.',color='black',markersize=12)
                    plt.plot(y_arr_nnan,x_fit_nnan,color='red')
                    plt.title('X-INTERPOLATION @ t='+str(int(t)))
                    plt.ylim(ed_l,xwidth-ed_u)
                    plt.show(block=False)
                ###################################################
            if CON==True:
                x_fit_full=gaus_params[i,t,:,1]
                
            if t%10==0:
                print '       -- Summing Aperture...'
            for j in range(0,n_rows):
                
                y_start=np.int(np.max([0,y0-extray]))

                sub_bkgd[t,j+y_start,:]=corr_sv[t,j+y_start,:]-bckgnd_sv[t,j+y_start,:]

                lower=np.nanmin([np.nanmax([ed_l,x_fit_full[j+y_start]-3*int(fwhm_av[i,t])]),xwidth-ed_u])
                upper=np.nanmax([np.nanmin([x_fit_full[j+y_start]+3*int(fwhm_av[i,t]),xwidth-ed_u]),ed_l])

                flat_spec[i,t,j+y_start]=np.nansum(sub_bkgd[t,j+y_start,int(lower):int(upper)])
                #removing extra from lower end and adding missing upper end... 
                extra_l=(lower-int(lower))*sub_bkgd[t,j+y_start,int(lower)]
                if upper>=(xwidth-1)-ed_u:
                    extra_u=0
                else:
                    extra_u=(upper-int(upper))*sub_bkgd[t,j+y_start,int(upper)+1]

                flat_spec[i,t,j+y_start]=flat_spec[i,t,j+y_start]+extra_u-extra_l

                #photon error is from original data, NOT corrected or bkgnd removed
                if upper>=(xwidth-1)-ed_u:
                    pht_err[i,t,j+y_start]=np.sqrt(np.nansum(obj_data[t,j,int(lower):int(upper)])
                                                         -(lower-int(lower))*obj_data[t,j,int(lower)])
                else:
                    pht_err[i,t,j+y_start]=np.sqrt(np.nansum(obj_data[t,j,int(lower):int(upper)])
                                                         +(upper-int(upper))*obj_data[t,j,int(upper)+1]
                                                         -(lower-int(lower))*obj_data[t,j,int(lower)])
                
            
        fig,ax=plt.subplots(2,2,figsize=(8,8))
        ax[0,0].plot(fwhm_av[i,:])
        plt.figtext(0.2,0.8,'fwhm_av',color='black')
            
        for j in range(0,n_rows):
            if j%100==0:
                ax[0,1].plot(gaus_params[i,:,j+y_start,1])
        plt.figtext(0.8,0.8,'x_centroid',color='black')
            
        for j in range(0,n_rows):
            if j%100==0:
                ax[1,0].plot(bkgd_params[i,:,j+y_start,0])
        plt.figtext(0.2,0.2,'bg slope',color='black')
                                     
        for j in range(0,n_rows):
            if j%100==0:
                ax[1,1].plot(bkgd_params[i,:,j+y_start,1])
        plt.figtext(0.8,0.2,'bg counts',color='black')
        plt.show(block=False)
        
        plt.figure(107)
        plt.clf()
        plt.cla()
        for t in range(0,n_exp):
            if t%10==0:
                plt.plot(np.linspace(0,2*ypixels/binny+ygap,2*ypixels/binny+ygap),flat_spec[i,t,:])
        plt.figtext(0.2,0.8,'OBJECT '+str(int(i)),fontsize=15,color='red')
        plt.xlabel('Stitched Pixels')
        plt.ylabel('ADUs')
        plt.show(block=False)
        
        print ' '
        time1=datetime.now()
        print'          time to run: ', time1-time0
                                               
        np.savez_compressed(SAVEPATH+'SpectraFitParams_'+str(int(i))+'.npz',sub_bkgd=sub_bkgd,bkgd=bckgnd_sv,corr=corr_sv)

    np.savez_compressed(SAVEPATH+'FlattenedSpectra.npz',flat_spec=flat_spec,fwhm_av=fwhm_av,gaus_params=gaus_params,
                            bkgd_params=bkgd_params,pht_err=pht_err)
                   
                    
                
                
                
    