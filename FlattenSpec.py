import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime

from setup import *

from outlier_removal import outlierr_c
from outlier_removal import outlierr

def Gaussian(x,a,b,c,d):
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))+d

def Fit_Gaussian(i,t,j,y_start,x,y,p0,gaus_params):
    p0=p0
    try:
        g_param,g_cov=curve_fit(Gaussian,x,y,p0=p0,maxfev=100000)
    except RuntimeError:
#        sub_bkgd[t,j,:]=y-background
        gaus_params[i,t,j+y_start,:]=gaus_params[i,t,j+y_start-1,:]
    else:
#        sub_bkgd[t,j,:]=y-background
        gaus_params[i,t,j+y_start,:]=g_param
    return gaus_params

def FlattenSpec(extray,SAVEPATH,ed_l,ed_u,binnx,binny,Lflat,Ldark,CON,ks_d,sig_d,ks_s,sig_s,ing_fwhm,
                ver_full,ver_a,ver_t,ver_x,ver_w,ver,data_corr,trip,time_start,time_trim,obj_skip,a_s,a_d):
    #extray= number of pixels in y direction extra that were extracted
    #SAVEPATH= location of saved 2D spec
    #filename= name of saved file
    #savefile= name to save as
    #corr= cosmic ray corecction?
    #ed_l= location of lower boundary between background/data
    #ed_u= location of upper boundary between background/data
    #ed_t= number of pixels on edges of 2D strip to trim
    #ks_d= kernel size for data area outlier detection
    #sig_d= sigma thershold for data outliers
    #ver_full = verbose output (LOTS OF PLOTS!)
    #ver_a  = plot x-aligned spectra every 10 exposures
    #ver_t = plot spectra every 10 frames
    #ver_x = plot x-shift fits
    #ver = corrections within ap
    #data_cor= run correction on gaussian?
    #trip= run median correction 3 times
    #a_s =  aperture size * fwhm_data
    
    ###########################################
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    
    
    if Ldark==True:
        dark_var=np.load(SAVEPATH+'Darks.npz')['var']
    if Lflat==True:
        flat_var=np.load(SAVEPATH+'Flats.npz')['var']
        
    gaus_params=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,4])*np.nan
    Xalign_data=np.empty([n_obj,n_exp,2*ypixels/binny+ygap,100/binnx+1])*np.nan
    fwhm_data=np.empty([n_obj,n_exp])
    flat_spec=np.empty([n_obj,n_exp,2*ypixels/binny+ygap])*np.nan
    
    for o in range(0,n_obj):
        if o in obj_skip:
            continue
       
        
        time0=datetime.now()
        print '-----------------'
        print '  OBJECT # ', o
        print '-----------------'
        
        print ' ----> loading data...'
        load=(np.load(SAVEPATH+'BG_SUBTRACTION_'+str(int(o))+'.npz'))  #n_exp,2*ypixels/binny+ygap,xwidth
        obj_data=load['corr']
        bkgd_dat=load['bkgd']
        bkgd_params=load['bkgd_params']
        print '      (done)'
        
        print ' ----> loading masks...'
        mask=(np.load(SAVEPATH+'FinalMasks.npz')['masks'])[o,:]
        print '      (done)'
                   
        
        y0=int(mask[1])  #pixel number of inital extraction
        #y_start=np.int(np.max([0,y0-extray]))  #including extray
        y_start=0  #background removal applies this shift 
        ### BINN THE SIZES OF THE MASKS in Y ###
        if binny>1:
            if y0<ypixels:
                y0=y0/binny
            if y0>ypixels:
                y0=(y0-ypixels-ygap)/binny+ypixels/binny+ygap
            
            if y_start<=ypixels:
                y_start=y_start/binny
            if y_start>ypixels:
                y_start=(y_start-ypixels-ygap)/binny+ypixels/binny+ygap
                
        ##################################
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        
        xpix_ar=np.linspace(1,xwidth,xwidth)
#         bkgd_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
#         corr_sv=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
    
#         sub_bkgd=np.empty([n_exp,2*ypixels/binny+ygap,xwidth])*np.nan
        
        #begin loop over time...
        for t in range(time_start,n_exp-time_trim):
            if t%10==0:
                print '       *** TIME: ',t,' ***'
            
            frame=np.array(obj_data[t,:,:])   #current frame
            
            #begin loop over pixels
            bad=0
            for j in range(0,n_rows):
                #print j, y_start, n_rows, j+y_start, 2*ypixels+ygap
                row_data=np.array(frame[j,:])
                if j>ypixels and j<=ypixels+ygap:  # if in gap
                    continue
                if not np.isfinite(row_data[0]):    # nans bookend the data
                    continue
                
                ## fitting gaussian
                g=np.argmax(row_data)
                if g>60/binnx and g<140/binnx:
                    p0=np.array([np.nanmax(row_data),g,ing_fwhm,0])
                else:
                    p0=np.array([np.nanmax(row_data),100/binnx,ing_fwhm,0])
                gaus_params=Fit_Gaussian(o,t,j,y_start,xpix_ar,row_data,p0,gaus_params)
                
                if np.nanmax(row_data)<=50 or gaus_params[o,t,j,1] < 60/binnx or gaus_params[o,t,j,1] > 140/binnx :
                    if ver_w==True:
                        print  ' --WARNING: LOW COUNTS OR UNEXPECTED CENTER POSITION (removing from fits) j'+str(j)+' t'+str(t)
                        bad+=1
                        if bad%10==0:
                            plt.figure(301,figsize=(15,2))
    #                         fig.subplots_adjust(wspace=0, hspace=0)
    #                         ax[0].plot(xpix_ar,obj_data[t,j-1,:],color='black',linewidth=5.0)
    #                         #ax[0].plot(xpix_ar,Gaussian(xpix_ar,*p0),linewidth=2.0,color='cyan')
    #                         ax[0].plot(xpix_ar,Gaussian(xpix_ar,*gaus_params[o,t,j-1,:]),linewidth=3.0,
    #                                    color='orange',linestyle='--')
    #                         plt.figtext(0.2,0.8,str(int(j-1)),color='black',fontsize=25)
                            plt.plot(xpix_ar,row_data,color='black',linewidth=5.0)
                            plt.plot(xpix_ar,Gaussian(xpix_ar,*p0),linewidth=2.0,color='cyan')
                            plt.plot(xpix_ar,Gaussian(xpix_ar,*gaus_params[o,t,j,:]),linewidth=3.0,color='red')
                            plt.figtext(0.15,0.7,str(int(j)),color='black',fontsize=25)
                            plt.show(block=False)
                            plt.close()
                    gaus_params[o,t,j,:]=np.empty([4])*np.nan
                    
                   
                
            # calculating fit to x-centers

            y_arr_nnan=np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap)[~np.isnan(gaus_params[o,t,:,1])]
            x_ctr_nnan=gaus_params[o,t,~np.isnan(gaus_params[o,t,:,1]),1]

            x_fit=np.polyfit(y_arr_nnan,medfilt(x_ctr_nnan,kernel_size=25),2)

            x_fit_nnan=(np.poly1d(x_fit))(y_arr_nnan)
            x_fit_full=(np.poly1d(x_fit))(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap))

            gaus_params[o,t,:,1]=x_fit_full
            
            if ver_x==True:
                if t%10==0:
                    plt.figure(104,figsize=(15,2))
                    plt.plot(y_arr_nnan,x_ctr_nnan,'.',color='black',markersize=8,alpha=0.4,zorder=0)
                    plt.plot(y_arr_nnan,medfilt(x_ctr_nnan,kernel_size=5),'.',color='black',markersize=8,zorder=1)
                    plt.plot(y_arr_nnan,x_fit_nnan,color='red',linewidth=2.0,alpha=0.5,zorder=3)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap),x_fit_full,color='red',zorder=4)
                    plt.title('X-INTERPOLATION @ t='+str(int(t)))
                    plt.ylim(40/binnx,160/binnx)
                    plt.show(block=False)
             
            ### correct within 1-a_s ##
            for j in range(0,n_rows):
                row_data=np.array(frame[j,:])
                if j>ypixels and j<=ypixels+ygap:  # if in gap
                    continue
                if not np.isfinite(row_data[0]):    # nans bookend the data
                    continue
                if not np.isfinite(gaus_params[o,t,j,2]): #if selected as a 'bad' row
                    continue
                fwhm=ing_fwhm
                reg_pix=np.append(xpix_ar[ed_l:int(gaus_params[o,t,j,1]-(a_d)*fwhm)],
                                  xpix_ar[int(gaus_params[o,t,j,1]+(a_d)*fwhm):int(xwidth-ed_u)])
                sub_g=row_data-Gaussian(xpix_ar,*gaus_params[o,t,j,:])
                reg_dat_0=np.append(sub_g[ed_l:int(gaus_params[o,t,j,1]-(a_d)*fwhm)],
                                  sub_g[int(gaus_params[o,t,j,1]+(a_d)*fwhm):int(xwidth-ed_u)])
                c1=0
                c2=0
                c3=0
                c1,reg_dat_1=outlierr_c(np.copy(reg_dat_0),ks_d,sig_d)
                c2,reg_dat_2=outlierr_c(np.copy(reg_dat_1),ks_d,sig_d)
                if trip==True:
                    c3,reg_dat_3=outlierr_c(np.copy(reg_dat_2),ks_d,sig_d)
                    reg_dat=np.copy(reg_dat_3)
                    tr=c1+c2+c3
                else:
                    reg_dat=np.copy(reg_dat_2)
                    tr=c1+c2

                reg_dat+=Gaussian(reg_pix,*gaus_params[o,t,j,:])
                reg_dat_0+=Gaussian(reg_pix,*gaus_params[o,t,j,:])
                reg_dat_1+=Gaussian(reg_pix,*gaus_params[o,t,j,:])
                reg_dat_2+=Gaussian(reg_pix,*gaus_params[o,t,j,:])
                if trip==True:
                    reg_dat_3+=Gaussian(reg_pix,*gaus_params[o,t,j,:])

                if ver==True:
                    if tr>1:
                        plt.figure(501,figsize=(14,4))
                        plt.title('BAKGROUND FILTERING: OBJ='+str(int(o))+
                                  ' TIME='+str(int(t))+' ROW='+str(np.round(100.*float(j)/float(n_rows),5))+'%')
                        plt.plot(xpix_ar,row_data,color='black',linewidth=6.0,zorder=1)
                        plt.plot(reg_pix,reg_dat_1,color='purple', linewidth=3.0,zorder=2)
                        plt.plot(reg_pix,reg_dat_2,color='blue',linewidth=3.0,zorder=3)
                        plt.figtext(0.15,0.7,c1,color='purple',fontsize=25)
                        plt.figtext(0.15,0.5,c2,color='blue',fontsize=25)
                        if trip==True:
                            plt.plot(reg_pix,reg_dat_3,color='green',linewidth=3.0,zorder=4)
                            plt.figtext(0.15,0.3,c3,color='green',fontsize=25)
                        plt.plot(xpix_ar,Gaussian(xpix_ar,*gaus_params[o,t,j,:]),linewidth=1.0,color='red',zorder=5)
                        plt.plot(reg_pix,Gaussian(reg_pix,*gaus_params[o,t,j,:]),linewidth=3.0,color='red',zorder=5)
                        plt.axvline(x=int(gaus_params[o,t,j,1]+(a_d)*fwhm),color='grey',linewidth=0.5)
                        plt.axvline(x=int(gaus_params[o,t,j,1]-(a_d)*fwhm),color='grey',linewidth=0.5)
                        plt.axvline(x=int(gaus_params[o,t,j,1]+(a_s)*fwhm),color='grey',linewidth=1.5)
                        plt.axvline(x=int(gaus_params[o,t,j,1]-(a_s)*fwhm),color='grey',linewidth=1.5)
                        #plt.ylim(-100,100)
                        plt.show(block=False)
                        plt.close()
                        
                row_data[ed_l:int(gaus_params[o,t,j,1]-(a_d)*fwhm)]=reg_dat[:int(gaus_params[o,t,j,1]-(a_d)*fwhm)-ed_l]   
                row_data[int(gaus_params[o,t,j,1]+(a_d)*fwhm):
                         int(xwidth-ed_u)]=reg_dat[int(gaus_params[o,t,j,1]-(a_d)*fwhm)-ed_l:]  
                frame[j,:]=row_data

            
            ### save data to X-align
            for j in range(0,n_rows):
                row_data=np.array(frame[j,:])
                if j>ypixels and j<=ypixels+ygap:  # if in gap
                    continue
                if not np.isfinite(row_data[0]):
                    continue
                #print (gaus_params[o,t,j+y_start,1]), int(gaus_params[o,t,j+y_start,1])
                #print int(gaus_params[o,t,j+y_start,1])-51, int(gaus_params[o,t,j+y_start,1])+51
#                 if np.isfinite(gaus_params[o,t,j,1])==False:
#                     gaus_params[o,t,j,1]=gaus_params[o,t,j-1,1]              
                #print o, t, j, gaus_params[o,t,j,:]
#                 if int(gaus_params[o,t,j,1])<=51:
#                     print len(row_data), int(gaus_params[o,t,j,1])
#                     arr=row_data[0:int(gaus_params[o,t,j,1])+50/binnx]
#                     Xalign_data[o,t,j,:(100/binnx+1-len(arr))]=arr
#                 if int(gaus_params[o,t,j,1])>len(row_data)-50:
#                     print len(row_data),int(gaus_params[o,t,j,1])
#                     arr=row_data[int(gaus_params[o,t,j,1])-(50/binnx)-1:]
#                     Xalign_data[o,t,j,(100/binnx+1-len(arr)):]=arr
                else:
                    Xalign_data[o,t,j,:]=row_data[int(gaus_params[o,t,j,1])-(50/binnx)-1:int(gaus_params[o,t,j,1])+50/binnx]
            
            if CON==False:
                fwhm_data[o,t]=int(np.nanmedian(gaus_params[o,t,:,2]))
                
            
            if ver_a==True:
                if t%10==0:
                    fig,ax=plt.subplots(2,1,figsize=(15,4))
                    fig.subplots_adjust(wspace=0, hspace=0)
                    ax[0].imshow((obj_data[t,:,:].T),cmap=plt.cm.plasma,aspect='auto')
                    ax[0].plot(gaus_params[o,t,:,1]-a_s*fwhm_data[o,t],color='white',linewidth=1.0)
                    ax[0].plot(gaus_params[o,t,:,1]+a_s*fwhm_data[o,t],color='white',linewidth=1.0)
                    ax[0].plot(gaus_params[o,t,:,1],color='white',linewidth=0.5,alpha=0.6)
                    #ax[0].set_ylim(100/binnx-3*a_s*fwhm_data[o,t],100/binnx+3*a_s*fwhm_data[o,t])
                    ax[0].set_xticklabels([])
                    ax[1].imshow((Xalign_data[o,t,:,:].T),cmap=plt.cm.plasma,aspect='auto')
                    ax[1].set_xlim(y_start,obj_data.shape[1]+y_start)
                    ax[1].axhline(y=51-a_s*fwhm_data[o,t],color='white',linewidth=1.0)
                    ax[1].axhline(y=51+a_s*fwhm_data[o,t],color='white',linewidth=1.0)
                    ax[1].axhline(y=51,color='white',linewidth=0.5,alpha=0.6)
                    ax[1].set_ylim(51-2*a_s*fwhm_data[o,t],51+2*a_s*fwhm_data[o,t])
                    ax[1].set_xticklabels([])
                    plt.figtext(0.15,0.70,t,color='grey',fontsize=30)
                    plt.show(block=False)
                    plt.close()
                    
        if CON==True:
            fwhm_data[o,:]=int(np.nanmedian(gaus_params[o,:,:,2]))
        
        ### flatten spectra ####
        rm_s=0
        flat_spec[o,:,:]=np.nansum(Xalign_data[o,:,:,:],axis=2)
        for t in range(0,n_exp):
            rm_s_a,flat_spec[o,t,:]=outlierr_c(np.copy(flat_spec[o,t,:]),ks_s,sig_s)
            rm_s+=rm_s_a
        
        if ver_t==True:
            norm=matplotlib.colors.Normalize(vmin=0,vmax=n_exp)
            colors=matplotlib.cm.viridis
            scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
            scal_m.set_array([])
            
            fig=plt.figure(301,figsize=(15,4))
            for t in range(0,n_exp):
                plt.plot(flat_spec[o,t,:],color=scal_m.to_rgba(t),linewidth=1.0)
            # [left, bottom, width, height
            cbaxes = fig.add_axes([0.15, 0.2, 0.02, 0.6]) 
            cb = plt.colorbar(scal_m, cax = cbaxes)  
            plt.figtext(0.2,0.8,rm_s,color='black',fontsize=25)
            plt.show(block=False)
            plt.close()
            
        print datetime.now()-time0   
    np.savez_compressed(SAVEPATH+'FlattenedSpectra.npz',flat_spec=flat_spec,
                        xalign=Xalign_data,fwhm_ar=fwhm_data,gaus_params=gaus_params)
     