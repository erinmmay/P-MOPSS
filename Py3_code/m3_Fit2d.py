import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib

from datetime import datetime

from setup import *

from mF_outlier_removal import outlierr_c
from mF_outlier_removal import outlierr_model

def Gaussian_P(x,*fit_params):
    a,b,c = fit_params
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))

# def Gaussian_P_bckgrnd(x,*fit_params):
#     a,b,c = fit_params[:3]
#     polyparams = fit_params[3:]
#     return (np.poly1d(polyparams))(x-b)

def Fit2d(SAVEPATH, extray, binnx, binny, fb, ed_l_in, ed_u_in, ed_t, 
          LOUTb, ks_b, sig_b,bg_order,bg_smooth, 
          LOUTd, ks_d, sig_d, 
          ver_full, ver_fit, ver_xcen, 
          time_start,time_trim, ing_fwhm, obj_skip):
    
    #SAVEPATH= location of saved 2D spec
    #extray= number of pixels in y direction extra that were extracted
    #ed_l= location of lower boundary between background/data - only used for initial guess of centroid
    #ed_u= location of upper boundary between background/data - only used for initial guess of centroid
    #ed_t= number of pixels on edges of 2D strip to trim - only used for initial guess of centroid
    #binnx= binning in x direction
    #binny= binning in y direction
    
    PARAMS=[extray,ed_l_in,ed_u_in,ed_t,binnx,binny, 
            LOUTb, ks_b, sig_b, bg_order,
            LOUTd, ks_d, sig_d, 
            ver_full, ver_fit, ver_xcen, 
            time_start,time_trim, obj_skip]
    
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    print(n_obj,n_exp)
    
    fit_params=np.empty([n_obj,n_exp,int(2*ypixels/binny+ygap),int(3+bg_order+1)])*np.nan
    fwhm_data=np.empty([n_obj,n_exp])*np.nan
    
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
        
        y0=int(mask[1])  
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
        
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        print(i, n_rows,xwidth,y_start)
        
        xpix_ar=np.linspace(1,xwidth,xwidth)
        
        bkgd_sv=np.empty([n_exp,int(2*ypixels/binny+ygap),int(xwidth)])*np.nan
        sub_bkgd=np.empty([n_exp,int(2*ypixels/binny+ygap),int(xwidth)])*np.nan
        x_fit_full = np.empty([n_exp,int(2*ypixels/binny+ygap)])*np.nan
        
        for t in range(time_start,n_exp-time_trim):
            treplaceb = 0
            treplaced = 0
            if t%10==0:
                print('       *** TIME: ',t,' ***')
            
            frame=np.copy(obj_data[t,:,:])   #current frame
            plt_data=np.empty([int(2*ypixels/binny+ygap),int(xwidth)])*np.nan
            
            ################# VERBOSE OUTPUT #################
            if ver_full==True:
                if t%10==0:
                    plt.figure(102,figsize=(15,4))
                    plt.title('RAW: OBJ='+str(int(i))+' TIME='+str(int(t)))
                    for j in range(n_rows-1,0,-1):
                        plt.plot(xpix_ar,frame[j,:],linewidth=2.0,alpha=0.2)
                    plt.axvline(x=ed_l_in, color='grey',linewidth=0.5)
                    plt.axvline(x=xwidth-ed_u_in,color='grey',linewidth=0.5)
                    plt.axvline(x=ed_t, color='grey',linewidth=0.5)
                    plt.axvline(x=xwidth-ed_t,color='grey',linewidth=0.5)
                    plt.xlim(0,xwidth)
                    plt.show(block=False)
                    plt.close()
                else:
                    print('*** ', t, ' ***')
            ###################################################
            
            # start loop over rows to do fits/ outliers
            for j in range(0,n_rows):
                if j+y_start>=ypixels/binny and j+y_start<ypixels/binny+ygap:
                    continue
                row_data=np.copy(frame[j,:])  #obj data is not y_start shifted 
                
                #check S/N of row:
                if (np.nanmax(row_data[int(ed_l_in):int(xwidth)-int(ed_u_in)])
                 < 2.0*np.sqrt(np.nanmedian(row_data[int(ed_t):int(ed_l_in)]))):
                    frame[j,:]=np.empty([len(row_data)])*np.nan
                    if t%10==0:
                        print('        LOW S/N AT ROW', j+y_start)
                    continue
                
                a_guess = (np.nanmax(row_data[int(ed_l_in):int(xwidth-ed_u_in)])-np.nanmedian(row_data[int(ed_t):int(ed_l_in)]))
                x0_guess = xpix_ar[np.argmax(row_data[int(ed_l_in):int(xwidth-ed_u_in)])]+ed_l_in
                ed_l = x0_guess - 3.0*ing_fwhm
                ed_u = xwidth-(x0_guess + 3.0*ing_fwhm)
                
                #initial p0 guess for Gaussian_P fit
                p0=np.array([a_guess,
                              x0_guess,
                             ing_fwhm])
                
                p0_bg = np.ones([int(bg_order+1)])
                #p0[-1] = np.nanmedian(np.append(row_data[int(ed_t):int(ed_l)],row_data[int(xwidth-ed_u):int(xwidth-ed_t)]))

                
                try:
                    # median filter background
                    x_left = np.copy(xpix_ar[int(ed_t):int(ed_l)])
                    bg_left  = np.copy(row_data[int(ed_t):int(ed_l)])
                    bg_left  = medfilt(bg_left, kernel_size = bg_smooth)
                    
                    x_right = np.copy(xpix_ar[int(xwidth-ed_u):int(xwidth-ed_t)])
                    bg_right = np.copy(row_data[int(xwidth-ed_u):int(xwidth-ed_t)])
                    bg_right = medfilt(bg_right, kernel_size = bg_smooth)
                    
                    row_data[int(ed_t):int(ed_l)]               = np.copy(bg_left)
                    row_data[int(xwidth-ed_u):int(xwidth-ed_t)] = np.copy(bg_right)
                    
                    nxpix_arb = np.append(x_left, x_right)
                    nrow_datab = np.append(bg_left, bg_right)
                    
                    nxpix_ard = xpix_ar[int(ed_l):int(xwidth-ed_u)]
                    nrow_datad = row_data[int(ed_l):int(xwidth-ed_u)]
                    
                    bg_param = np.polyfit(nxpix_arb-xwidth/2.,nrow_datab, bg_order)
                    bg_fit = np.poly1d(bg_param)

                    g_param,g_cov=curve_fit(Gaussian_P, 
                                            nxpix_ard, 
                                            nrow_datad - bg_fit(nxpix_ard-xwidth/2.), 
                                            p0=p0,maxfev=100000)
                
                    fit_params[i,t,int(j+y_start),:3] = g_param
                    fit_params[i,t,int(j+y_start),3:] = bg_param
                    
                except RuntimeError:
                    fit_params[i,t,int(j+y_start),:] = fit_params[i,t,int(j+y_start-1),:]
                
                else:
                    fit_params[i,t,int(j+y_start),:3] = g_param
                    fit_params[i,t,int(j+y_start),3:] = bg_param
                    
                    
                ################# VERBOSE OUTPUT #################
                if ver_fit==True:
                    if t%10==0:
                        plt.figure(501,figsize=(15,6))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        plt.plot(xpix_ar,row_data,color='black',linewidth=4.0)

                        plt.plot(xpix_ar,Gaussian_P(xpix_ar,*p0),
                                   color='blue',linewidth=1.0,linestyle='--')
                        plt.plot(xpix_ar,
                                 Gaussian_P(xpix_ar,*fit_params[i,t,j+y_start,:3])+
                                 bg_fit(xpix_ar-xwidth/2.),
                                   color='cyan',linewidth=2.0)
                        plt.axvline(x=fit_params[i,t,j+y_start,1],color='red',linewidth=2.0)
                        plt.axvline(x=fit_params[i,t,j+y_start,1]-3.*2.*np.sqrt(2.*np.log(2.))*fit_params[i,t,j+y_start,2],
                                      color='tomato',linewidth=1.0)
                        plt.axvline(x=fit_params[i,t,j+y_start,1]+3.*2.*np.sqrt(2.*np.log(2.))*fit_params[i,t,j+y_start,2],
                                      color='tomato',linewidth=1.0) 
                        plt.xlim(0,xwidth)
                        plt.figtext(0.2,0.75,str(int(i))+' '+str(int(t))+' '+str(int(j)),fontsize=20)
                        plt.axvline(x=ed_t,color='grey')
                        plt.axvline(x=ed_l,color='grey')
                        plt.axvline(x=xwidth-ed_u, color='grey')
                        plt.axvline(x=xwidth-ed_t, color='grey')
                        plt.show(block=False)
                        plt.close()
                ##################################################                    
                # background outlier removal
                replaced = []
                replaceb = []
                
                if LOUTb == True:
                    
                    
                    nxpix_arb=np.append(xpix_ar[int(ed_t):int(ed_l)],xpix_ar[int(xwidth-ed_u):int(xwidth-ed_t)])
                    nrow_datab=np.append(row_data[int(ed_t):int(ed_l)],row_data[int(xwidth-ed_u):int(xwidth-ed_t)])

                    model=(Gaussian_P(nxpix_ard,*fit_params[i,t,int(j+y_start),:3])
                           + bg_fit(nxpix_ard-xwidth/2.))
                 
                    median=medfilt(nrow_datab,kernel_size=ks_b)
                    #delta = nrow_datab - median
                    row_saveb, lowcut, upcut= sigmaclip(nrow_datab , low = sig_b, high=sig_b)
                    
                   
                    replaceb = np.append(np.where(nrow_datab <= lowcut)[0], np.where(nrow_datab >= upcut)[0])
                    #print(i, t, j, nrow_datab, replace)
                    #row_saveb += median
                    nrow_dataB = np.copy(nrow_datab)
                    nrow_dataB[replaceb]=median[replaceb]
                    
                    treplaceb += len(replaceb)
                    
                    lena = len(xpix_ar[int(ed_t):int(ed_l)])
                    lenb = len(xpix_ar[int(xwidth-ed_u):int(xwidth-ed_t)])
                    row_data[int(ed_t):int(ed_l)]=np.copy(nrow_dataB[:lena])
                    row_data[int(xwidth-ed_u):int(xwidth-ed_t)]=np.copy(nrow_dataB[lena:])
                    
                        
                if LOUTd == True:
                    
#                     nxpix_ard=np.copy(xpix_ar[int(ed_l):int(xwidth-ed_u)])
#                     nrow_datad=np.copy(row_data[int(ed_l):int(xwidth-ed_u)])

                    model=(Gaussian_P(nxpix_ard,*fit_params[i,t,int(j+y_start),:3])
                           + bg_fit(nxpix_ard-xwidth/2.))
                    
                    median=medfilt(nrow_datad,kernel_size=ks_d)
                    #delta = nrow_datad - median
                    row_saved, lowcut, upcut = sigmaclip(nrow_datad , low = sig_d, high=sig_d)
                                                 
                    replaced = np.append(np.where(nrow_datad <= lowcut)[0], np.where(nrow_datad >= upcut)[0])
                    #print(i, t, j, replace)
                    #row_saved += median
                    nrow_dataD = np.copy(nrow_datad)
                    nrow_dataD[replaced]=median[replaced]
                    
                    treplaced += len(replaced)
                    
                    row_data[int(ed_l):int(xwidth-ed_u)]=np.copy(nrow_dataD)

                        
                if LOUTd == True or LOUTb == True:
                    ### refit- model ###
                    ## FIT IN COMBINATION -- polynomial and gaussian ##
                    p0=fit_params[i,t,int(j+y_start),:3]
                    p0_bg = fit_params[i,t,int(j+y_start),3:]
                    
                    try:
                        # median filter background
                        x_left = np.copy(xpix_ar[int(ed_t):int(ed_l)])
                        bg_left  = np.copy(row_data[int(ed_t):int(ed_l)])
                        bg_left  = medfilt(bg_left, kernel_size = bg_smooth)

                        x_right = np.copy(xpix_ar[int(xwidth-ed_u):int(xwidth-ed_t)])
                        bg_right = np.copy(row_data[int(xwidth-ed_u):int(xwidth-ed_t)])
                        bg_right = medfilt(bg_right, kernel_size = bg_smooth)

                        row_data[int(ed_t):int(ed_l)]               = np.copy(bg_left)
                        row_data[int(xwidth-ed_u):int(xwidth-ed_t)] = np.copy(bg_right)

                        nxpix_arb = np.append(x_left, x_right)
                        nrow_datab = np.append(bg_left, bg_right)

                        nxpix_ard = xpix_ar[int(ed_l):int(xwidth-ed_u)]
                        nrow_datad = row_data[int(ed_l):int(xwidth-ed_u)]

                        bg_param = np.polyfit(nxpix_arb-xwidth/2.,nrow_datab, bg_order)
                        bg_fit = np.poly1d(bg_param)

                        g_param,g_cov=curve_fit(Gaussian_P, 
                                                nxpix_ard, 
                                                nrow_datad - bg_fit(nxpix_ard-xwidth/2.), 
                                                p0=p0,maxfev=100000)

                        fit_params[i,t,int(j+y_start),:3] = g_param
                        fit_params[i,t,int(j+y_start),3:] = bg_param
                        
                    except RuntimeError:
                        fit_params[i,t,int(j+y_start),:]=fit_params[i,t,int(j+y_start-1),:]
                    else:
                        fit_params[i,t,int(j+y_start),:3] = g_param
                        fit_params[i,t,int(j+y_start),3:] = bg_param
                    
                
                ################# VERBOSE OUTPUT #################
                if ver_fit==True:
                    if t%10==0:
#                         if len(replaceb)<10 or len(replaced)==0:
#                             continue
                        print(p0)
                        print(fit_params[i,t,int(j+y_start),:])
        
                        plt.figure(505,figsize=(15,6))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        if LOUTd == True:
                            plt.plot(nxpix_ard,nrow_datad,color='grey',linewidth=2.0)
                        if LOUTb == True:
                            plt.plot(nxpix_arb,nrow_datab,color='grey',linewidth=2.0)
                        plt.plot(xpix_ar,row_data,color='black',linewidth=4.0)
                        plt.plot(xpix_ar,(Gaussian_P(xpix_ar,*fit_params[i,t,j+y_start,:3])
                                          +bg_fit(xpix_ar-xwidth/2.)),
                                   color='cyan',linewidth=2.0)
                        
                        plt.axvline(x=fit_params[i,t,int(j+y_start),1],color='red',linewidth=2.0)
                        
                        plt.axvline(
                            x=fit_params[i,t,int(j+y_start),1]-3.*2.*np.sqrt(2.*np.log(2.))*fit_params[i,t,int(j+y_start),2],
                                      color='tomato',linewidth=1.0)
                        plt.axvline(
                            x=fit_params[i,t,int(j+y_start),1]+3.*2.*np.sqrt(2.*np.log(2.))*fit_params[i,t,int(j+y_start),2],
                                      color='tomato',linewidth=1.0) 
                        plt.xlim(0,xwidth)
                        if LOUTb == True:
                            plt.figtext(0.2,0.4,len(replaceb),fontsize=20,color='black')  
                        if LOUTd == True:
                            plt.figtext(0.2,0.3,len(replaced),fontsize=20,color='blue')
                        plt.axvline(x=ed_t,color='grey')
                        plt.axvline(x=ed_l,color='grey')
                        plt.axvline(x=xwidth-ed_u, color='grey')
                        plt.axvline(x=xwidth-ed_t, color='grey')
                            
                        plt.show(block=False)
                        plt.close(505)
                           
                            
                        ### background plot #
                        plt.figure(506,figsize=(15,6))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        plt.plot(nxpix_arb,nrow_datab,color='black',linewidth=4.0)
                        

                        plt.plot(xpix_ar,bg_fit(xpix_ar-xwidth/2.),
                                   color='cyan',linewidth=2.0)
                        
                        plt.axvline(x=fit_params[i,t,int(j+y_start),1],color='red',linewidth=2.0)
                        
                        plt.axvline(x=ed_t,color='grey')
                        plt.axvline(x=ed_l,color='grey')
                        plt.axvline(x=xwidth-ed_u, color='grey')
                        plt.axvline(x=xwidth-ed_t, color='grey')
                        plt.xlim(0,xwidth)
                        #plt.ylim(0,5000)
                        plt.show(block=False)
                        plt.close(506)
                ##################################################    
                    
#                 bkgrnd_params = np.copy((fit_params[i,t,j+y_start,:]))
#                 bkgrnd_params[:3] = 1.0
#                 bkgrnd_params[0] = 0.0
                bkgd_sv[t,j+y_start,:]=bg_fit(xpix_ar-xwidth/2.)
                sub_bkgd[t,int(j+y_start),:]=np.copy(row_data-bkgd_sv[t,int(j+y_start),:])
                
                frame[j,:]=row_data
                plt_data[int(j+y_start),:]=np.copy(sub_bkgd[t,int(j+y_start),:])
                
            
            #calculating fit to x-centers
            y_arr_nnan=np.linspace(1,2*ypixels/binny+ygap,
                                   2*ypixels/binny+ygap)[~np.isnan(fit_params[i,t,:,1])]
            x_ctr_nnan=fit_params[i,t,~np.isnan(fit_params[i,t,:,1]),1]

            x_fit=np.polyfit(y_arr_nnan,medfilt(x_ctr_nnan,kernel_size=25),2)

            x_fit_nnan=(np.poly1d(x_fit))(y_arr_nnan)
            x_fit_full[t,:]=(np.poly1d(x_fit))(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap))

            #fit_params[i,t,:,1]=x_fit_full
            
            fwhm_data[i,t]=2.*np.sqrt(2.*np.log(2.))*np.nanmedian(fit_params[i,t,:,2])
            #print fwhm_data[i,:]
            
            if ver_xcen==True:
                if t%10==0:
                    if np.nanmin(plt_data)<0:
                        plt_data += np.nanmean(bkgd_sv)
                    pltymin=np.nanmin(x_fit_nnan-6.*fwhm_data[i,t])
                    pltymax=np.nanmax(x_fit_nnan+6.*fwhm_data[i,t])
                    #print(pltymin,pltymax)
                    vmin = np.nanmin(plt_data[:,int(pltymin):int(pltymax)])
                    vmax = np.nanmax(plt_data[:,int(pltymin):int(pltymax)])
                    #print(np.nanmin(plt_data),np.nanmax(plt_data))
                    #print(vmin,vmax)
                    plt.figure(104,figsize=(14,8))
                    plt.imshow((plt_data.T),cmap=plt.cm.plasma,aspect='auto',origin='lower',
                               extent=(0, 2*ypixels/binny+ygap, 0, xwidth),
                               norm=colors.LogNorm(vmin,vmax))
                    plt.plot(y_arr_nnan,x_fit_nnan,color='white',linewidth=4.0,zorder=4)
                    plt.plot(np.linspace(1,2*ypixels/binny+ygap,2*ypixels/binny+ygap), 
                             fit_params[i,t,:,1],color='grey',linewidth=4.0,zorder=3)
                    
                    plt.plot(y_arr_nnan,x_fit_nnan-1.*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.plot(y_arr_nnan,x_fit_nnan+1.*fwhm_data[i,t],color='white',linewidth=0.5)
                   
                    plt.plot(y_arr_nnan,x_fit_nnan-2.*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.plot(y_arr_nnan,x_fit_nnan+2.*fwhm_data[i,t],color='white',linewidth=0.5)
                    
                    plt.plot(y_arr_nnan,x_fit_nnan-3.*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.plot(y_arr_nnan,x_fit_nnan+3.*fwhm_data[i,t],color='white',linewidth=0.5)
                    
                    plt.plot(y_arr_nnan,x_fit_nnan-4.*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.plot(y_arr_nnan,x_fit_nnan+4.*fwhm_data[i,t],color='white',linewidth=0.5)
                    
                    plt.plot(y_arr_nnan,x_fit_nnan-5.*fwhm_data[i,t],color='white',linewidth=0.5)
                    plt.plot(y_arr_nnan,x_fit_nnan+5.*fwhm_data[i,t],color='white',linewidth=0.5)
                    
                    plt.figtext(0.2,0.7,str(int(t)),fontsize=20,color='yellow')
                    
                    plt.ylim(np.nanmin(x_fit_nnan-6.*fwhm_data[i,t]),
                             np.nanmax(x_fit_nnan+6.*fwhm_data[i,t]))
                    
                    plt.show(block=False)
                    plt.close()
            
            ################# VERBOSE OUTPUT #################
            if ver_full==True:
                if t%10==0:
                    plt.figure(102,figsize=(14,4))
                    plt.title('CORRECTED: OBJ='+str(int(i))+' TIME='+str(int(t)))
                    for j in range(n_rows-1, 0, -1):
                        plt.plot(xpix_ar,sub_bkgd[t,j+y_start,:],linewidth=1.0,alpha=0.2)
                    if LOUTb == True:
                        plt.figtext(0.2,0.4,str(treplaceb) +'    ' + str(np.round(treplaceb/n_rows,3)),fontsize=20,color='black') 
                    if LOUTd == True:
                        plt.figtext(0.2,0.3,str(treplaced) +'    ' + str(np.round(treplaced/n_rows,3)),fontsize=20,color='blue')   
                    plt.show(block=False)
                    plt.close()
                if t%10==0:
                    plt.figure(103,figsize=(14,4))
                    plt.title('CORRECTED: OBJ='+str(int(i))+' TIME='+str(int(t)))
                    for j in range(n_rows-1, 0, -1):
                        plt.plot(xpix_ar,sub_bkgd[t,j+y_start,:],linewidth=1.0,alpha=0.2)
                    if LOUTb == True:
                        plt.figtext(0.2,0.4,str(treplaceb) +'    ' + str(np.round(treplaceb/n_rows,3)),fontsize=20,color='black') 
                    if LOUTd == True:
                        plt.figtext(0.2,0.3,str(treplaced) +'    ' + str(np.round(treplaced/n_rows,3)),fontsize=20,color='blue')   
                    plt.axhline(y=0.0,color='black',lw=2.0,ls='--')
                    plt.axvline(x=ed_t, color='black', lw = 2.0, ls='--')
                    plt.axvline(x=xwidth-ed_t, color='black', lw = 2.0, ls='--')
                    plt.ylim(-200,200)
                    plt.show(block=False)
                    plt.close()
            ###################################################
            
            
        print(datetime.now()-time0) 
        np.savez_compressed(SAVEPATH+'2DFit_Obj'+str(int(i))+'.npz',
                            params=PARAMS, background = bkgd_sv, data = sub_bkgd, 
                            fwhm=fwhm_data[i,:], gaus=fit_params[i,:,:,:], x_fit = x_fit_full)
                    
                    
    