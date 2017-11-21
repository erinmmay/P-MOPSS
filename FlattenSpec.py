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

def FlattenSpec(ex,SAVEPATH,corr):
    
    n_obj=int(np.load(SAVEPATH+'FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load(SAVEPATH+'HeaderData.npz')['n_exp']
    
    flat_spec=np.empty([n_obj,n_exp,2*ypixels+ygap])*np.nan
    pht_err=np.zeros_like(flat_spec)
    tot_err=np.zeros_like(flat_spec)
    
    dark_var=np.load(SAVEPATH+'Darks.npz')['var']
    flats_var=np.load(SAVEPATH+'Flats.npz')['var']

    for i in range(0,n_obj):
        time0=datetime.now()
        print '-----------------'
        print '  OBJECT # ', i
        print '-----------------'
        if corr==True:
            obj_data=(np.load(SAVEPATH+'Corrected'+str(int(i))+'.npz'))['data']
        if corr==False:
            obj_data=(np.load(SAVEPATH+'2DSpec_obj'+str(int(i))+'.npz'))['data']
        sub_bkgd=np.zeros_like(obj_data)*np.nan
        mask=(np.load(SAVEPATH+'FinalMasks.npz')['masks'])[i,:]
        y0=int(mask[1])
        y_start=np.int(np.max([0,y0-ex]))
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        xpix_ar=np.linspace(1,xwidth,xwidth)
        fwhm_ar=np.empty([n_exp,2*ypixels+ygap])*np.nan
        fwhm_av=np.empty([n_exp])*np.nan
        cent_ar=np.empty([n_exp,2*ypixels+ygap])*np.nan
        bckgrnd=np.empty([2,n_exp,2*ypixels+ygap])*np.nan
        
        for t in range(0,n_exp):
            if t%10==0:
                print '    -->> TIME: ',t
                print '       -- FITTING GAUSSIANS'
            frame=obj_data[t,:,:]
            
            ed_l=85
            ed_u=88
            sd=2
            ms=5  # MUST BE GREATER THAN SD
            ks=29#5
            
            ms_a=15
            ks_a=7#5

            plt.figure(102,figsize=(14,4))
            plt.clf()
            plt.cla()
            plt.title('RAW: OBJ='+str(int(i))+' TIME='+str(int(t)))
            for j in range(0,n_rows):
                plt.plot(xpix_ar,frame[j,:],linewidth=1.0)
            plt.axvline(x=ed_l, color='grey',linewidth=0.5)
            plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
            plt.axvline(x=sd, color='grey',linewidth=0.5)
            plt.axvline(x=xwidth-sd,color='grey',linewidth=0.5)
            plt.show(block=False)
            
            for j in range(0,n_rows):
                row_data=frame[j,:]
                if not np.isfinite(row_data[0]):
                    continue
               
                bg_pix=np.append(xpix_ar[sd:ed_l],xpix_ar[xwidth-ed_u:xwidth-sd])
                bg_dat=np.append(row_data[sd:ed_l],row_data[xwidth-ed_u:xwidth-sd])
               
                
                #### doing background outlier rejection ###
                #print '       -- (outlier bg pixel rejection)'
                counter=0
                bg_dat_meda=medfilt(bg_dat,kernel_size=ks)
                
                dat_medfta=medfilt(row_data,kernel_size=ks_a)
                
                #plt.figure(105,figsize=(3.,3.))
                #plt.clf()
                #plt.cla()
                #plt.plot(bg_pix,bg_dat,color='black',linewidth=2.0)
                #plt.plot(bg_pix,bg_dat_med,color='cyan',linewidth=1.0)
                #plt.title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                #plt.show(block=False)
                
                dat_medft=row_data-dat_medfta
                dat_std=np.nanstd(dat_medft)
                dat_med=np.nanmedian(dat_medft)
                
                bg_dat_med=bg_dat-bg_dat_meda
                bg_std=np.nanstd(bg_dat_med)
                bg_med=np.nanmedian(bg_dat_med)
                

                        
               
               
                for b in range(0,len(bg_pix)):
                    b=int(b)
                    if bg_dat_med[b]>3.0*bg_std+bg_med or bg_dat_med[b]<bg_med-3.0*bg_std:
                        counter+=1
                        val=bg_dat_meda[b]#bg_dat[b]-bg_dat_med[b]#np.nanmedian(np.append(bg_dat[minx:b-ms],bg_dat[b+1+ms:maxx]))
                        ind=np.where(xpix_ar==bg_pix[b])
                        bg_dat[b]=val
                        row_data[ind]=val
                        
                #for b in range(sd,len(xpix_ar)-sd):
                #    if b>sd+20 and b<xwidth-sd-20:
                #        continue
                #    b=int(b)
                #    if dat_medft[b]>3.0*dat_std+dat_med or dat_medft[b]<dat_med-3.0*dat_std:
                #        counter+=1
                #        
                #        minx=np.nanmax([sd,b-ms])
                #        maxx=np.nanmin([b+ms,xwidth-sd])
                       
                #        val=dat_medfta[b]#np.nanmedian(np.append(row_data[minx:b-sd],row_data[b+1+sd:maxx]))
                        #ind=np.where(xpix_ar==bg_pix[b])
                        #bg_dat[b]=val
                #        row_data[b]=val
                        
                        #plt.figure(105,figsize=(3.,3.))
                        #plt.clf()
                        #plt.cla()
                        #plt.plot(xpix_ar,row_dat,color='black',linewidth=2.0)
                        #plt.plot(a_pix,gfit,color='cyan',linewidth=1.0)
                        ##plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                        #plt.title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                        #plt.show(block=False)
                        
                        #plt.figure(106,figsize=(8.,3.))
                        #plt.clf()
                        #plt.cla()
                        #plt.plot(xpix_ar[minx:maxx],dat_medft[minx:maxx],color='black',linewidth=2.0)
                        #plt.plot(xpix_ar[minx:maxx],(row_data-dat_medfta)[minx:maxx],color='red',linewidth=1.0)
                        #plt.axhline(y=dat_med,color='lime')
                        #plt.axhline(y=dat_med+3*dat_std,color='lime',linestyle='--')
                        #plt.axhline(y=dat_med-3*dat_std,color='lime',linestyle='--')
                        ##plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                        #plt.title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                        #plt.show(block=False)
                        
               
                  
                if (counter/xwidth)*100.>5.:
                    print 'CAUTION: MORE THAN 5% CORRECTED FOR ROW ',j, '!!!!!'
                obj_data[t,j,:]=row_data   #updating with background fix
                frame[j,:]=row_data
                
                bg_params=np.polyfit(bg_pix,bg_dat,1)
                background=(np.poly1d(bg_params))(xpix_ar)
                
                #if j%100==0:
                #    plt.figure(100)
                #    plt.plot(bg_pix,bg_dat,color='magenta',linewidth=0.5)
                #    plt.plot(xpix_ar,background,color='lime',linewidth=0.5,linestyle='--')
                #    plt.show(block=False)
                    #plt.pause(0.5)
                
                bckgrnd[:,t,j+y_start]=bg_params
                p0=np.array([np.nanmax(row_data),np.argmax(row_data),10,background[50]])
                
                try:
                    g_param,g_cov=curve_fit(Gaussian,xpix_ar,row_data,p0=p0,maxfev=10000)
                except RuntimeError:
                    sub_bkgd[t,j,:]=np.empty([len(row_data)])*np.nan
                else:
                    fwhm_ar[t,j+y_start]=2*np.sqrt(2*np.log(2))*g_param[2]
                    cent_ar[t,j+y_start]=int(g_param[1])
                    sub_bkgd[t,j,:]=row_data-background
                 
                
                #### outliers along gaussian ####
                #print '       -- (outlier dat pixel rejection)'
                
                a_pix=xpix_ar[ed_l:xwidth-ed_u]
                a_dat=row_data[ed_l:xwidth-ed_u]
                
                gfit=Gaussian(a_pix,*g_param)
                counter=0
                
                a_median=medfilt(a_dat,kernel_size=ks_a)
                
                a_dat_med=a_dat-gfit
                a_std=np.nanstd(a_dat_med)
                a_med=np.nanmedian(a_dat_med)
   
                
                for b in range(0,len(a_pix)):
                    b=int(b)
                    olv=3.5
                    if a_dat_med[b]>olv*a_std+a_med or a_dat_med[b]<a_med-olv*a_std:
                        counter+=1
                        
                        fig,ax=plt.subplots(1,3,figsize=(9.,2.))
                        
                        ax[0].plot(a_pix,a_dat,color='black',linewidth=2.0)
                        ax[0].plot(a_pix,gfit,color='cyan',linewidth=1.0)
                        #plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                        ax[0].set_title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                        
                        val=a_median[b]#a_dat[b]-a_dat_med[b]#np.nanmedian(np.append(bg_dat[minx:b-ms],bg_dat[b+1+ms:maxx]))
                        ind=np.where(xpix_ar==a_pix[b])
                        a_dat[b]=val
                        row_data[ind]=val
                #  

                      
                #        
                        ax[1].plot(a_pix,a_dat_med,color='black',linewidth=2.0)
                        ax[1].plot(a_pix,a_dat-gfit,color='red',linewidth=1.0)
                        ax[1].axhline(y=a_med,color='lime')
                        ax[1].axhline(y=a_med+olv*a_std,color='lime',linestyle='--')
                        ax[1].axhline(y=a_med-olv*a_std,color='lime',linestyle='--')
                        #plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                        ax[1].set_title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                        
                        
                        bckgrnd[:,t,j+y_start]=bg_params
                        p0=g_param
                
                        try:
                            g_param,g_cov=curve_fit(Gaussian,xpix_ar,row_data,p0=p0,maxfev=10000)
                        except RuntimeError:
                            sub_bkgd[t,j,:]=np.empty([len(row_data)])*0.0
                        else:
                            fwhm_ar[t,j+y_start]=2*np.sqrt(2*np.log(2))*g_param[2]
                            cent_ar[t,j+y_start]=int(g_param[1])
                            sub_bkgd[t,j,:]=row_data-background
                            
                        gfit=Gaussian(a_pix,*g_param)
                        a_dat=row_data[ed_l:xwidth-ed_u]
                       
                            
                        ax[2].plot(a_pix,a_dat,color='black',linewidth=2.0)
                        ax[2].plot(a_pix,gfit,color='cyan',linewidth=1.0)
                        #plt.plot(a_pix,a_dat_med,color='cyan',linewidth=1.0)
                        ax[2].set_title(str(int(i))+' '+str(int(t))+' '+str(int(j)))
                        
                        plt.show(block=False)
                        
                if (counter/xwidth)*100.>5.:
                    print 'CAUTION: MORE THAN 5% CORRECTED FOR ROW ',j, '!!!!!'
                
                    #if t%10==0 and j%100==0:
                    #    plt.figure(1)
                    #    plt.clf()
                    #    plt.cla()
                    #    plt.plot(xpix_ar,row_data,color='black',linewidth=4.0)
                    #    plt.plot(xpix_ar,Gaussian(xpix_ar,*g_param),color='orange',linewidth=2.0,linestyle='-')
                    #    plt.plot(xpix_ar,background,color='blue',linewidth=2.0,linestyle='--')
                    #    plt.axvline(x=cent_ar[t,j],color='red',linewidth=1.5)
                    #    plt.axvline(x=cent_ar[t,j]-fwhm_ar[t,j], color='green',linestyle='--',linewidth=0.5)
                    #    plt.axvline(x=cent_ar[t,j]+fwhm_ar[t,j], color='green',linestyle='--',linewidth=0.5)
                    #    plt.axvline(x=cent_ar[t,j]-3.*fwhm_ar[t,j], color='darkgreen',linestyle='--',linewidth=1.5)
                    #    plt.axvline(x=cent_ar[t,j]+3.*fwhm_ar[t,j], color='darkgreen',linestyle='--',linewidth=1.5)
                    #    plt.figtext(0.1,0.9,str(int(i))+' '+str(int(t))+' '+str(int(j)))
                    #    plt.show(block=False)
            fwhm_av[t]=(np.nanmedian(fwhm_ar[t,y_start:y_start+n_rows]))
            
            plt.figure(103,figsize=(14,4))
            plt.clf()
            plt.cla()
            plt.title('FIXED: OBJ='+str(int(i))+' TIME='+str(int(t)))
            for j in range(0,n_rows):
                plt.plot(xpix_ar,frame[j,:],linewidth=1.0)
            plt.axvline(x=ed_l, color='grey',linewidth=0.5)
            plt.axvline(x=xwidth-ed_u,color='grey',linewidth=0.5)
            plt.axvline(x=sd, color='grey',linewidth=0.5)
            plt.axvline(x=xwidth-sd,color='grey',linewidth=0.5)
            plt.show(block=False)
            
            if t%10==0:
                print '       -- SUMMING APERTURE'
            for j in range(0,n_rows):
                row_data=frame[j,:]
                if not np.isfinite(row_data[0]):
                    continue
                low=np.nanmax([0,cent_ar[t,j]-3*int(fwhm_av[t])])
                up=np.nanmin([cent_ar[t,j]+3*int(fwhm_av[t]),xwidth])
                #print flat_spec.shape, '     ', i,t,j+y0,j,n_rows
                y_start=np.int(np.max([0,y0-ex]))
                flat_spec[i,t,j+y_start]=np.nansum(sub_bkgd[t,j,int(low):int(up)])
                pht_err[i,t,j+y_start]=np.sqrt(flat_spec[i,t,j+y_start])
                tot_err[i,t,j+y_start]=np.sqrt((pht_err[i,t,j+y_start])**2.+dark_var)
                #tot_err[i,t,j+ystart]=np.sqrt((tot_err[i,t,j+ystart]/flat_spec[i,t,j+y_start]
        plt.figure(2)
        plt.clf()
        plt.cla()
        for t in range(0,n_exp):
            if t%10==0:
                plt.plot(np.linspace(0,2*ypixels+ygap,2*ypixels+ygap),flat_spec[i,t,:])
        plt.figtext(0.2,0.8,'OBJECT '+str(int(i)),fontsize=15,color='red')
        plt.xlabel('Stitched Pixels')
        plt.ylabel('ADUs')
        plt.show(block=False)
        print ' '
        time1=datetime.now()
        print'          time to run: ', time1-time0
        if corr==True:
            np.savez_compressed(SAVEPATH+'SpectraFitParams_'+str(int(i))+'_Corr.npz',fwhm=fwhm_ar,fwhm_av=fwhm_av,x=cent_ar,bg=bckgrnd)
        else:
            np.savez_compressed(SAVEPATH+'SpectraFitParams_'+str(int(i))+'.npz',fwhm=fwhm_ar,fwhm_av=fwhm_av,x=cent_ar,bg=bckgrnd)
    if corr==True:
            np.savez_compressed(SAVEPATH+'FlattenedSpectra_Corr.npz',flat_spec=flat_spec,fwhm_ar=fwhm_ar,fwhm_av=fwhm_av,cent_ar=cent_ar,pht_err=pht_err,tot_err=tot_err)
    else:
            np.savez_compressed(SAVEPATH+'FlattenedSpectra.npz',flat_spec=flat_spec,fwhm_ar=fwhm_ar,fwhm_av=fwhm_av,cent_ar=cent_ar,pht_err=pht_err,tot_err=tot_err)
    return flat_spec
        
    
