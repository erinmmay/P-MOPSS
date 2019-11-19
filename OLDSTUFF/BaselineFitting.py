import scipy

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from numpy.linalg import inv
from outlier_removal import outlierr


from setup import *

def blfit_white(SAVEPATH,order,avg,olow,ohigh,ybot,ytop,timein,timeeg,corr,time_trim,time_skip,outlier):

    order=order
    low=olow
    high=ohigh

    ybot=ybot
    ytop=ytop

    time0=np.load(SAVEPATH+'Obs_times.npz')['times']

    if corr==True:
        LC=np.load(SAVEPATH+'LCwhite_Corr.npz')['data']
        err_t=np.load(SAVEPATH+'LCwhite_Corr.npz')['err_t']
        err_p=np.load(SAVEPATH+'LCwhite_Corr.npz')['err_p']
    else:
        LC=np.load(SAVEPATH+'LCwhite.npz')['data']
        err_t=np.load(SAVEPATH+'LCwhite.npz')['err_t']
        err_p=np.load(SAVEPATH+'LCwhite.npz')['err_p']

#     for t in range(0,len(time0)):
#         if t>len(time0)-time_trim:
#             LC[t]=np.nan
           

 #     for f in range(0,len(LC)):
#         if LC[f]<low or LC[f]>high or np.isfinite(LC[f])==False:
#             if f>1 and f<len(LC)-1:
#                 LC[f]=np.nanmedian(np.append(LC[f-1],LC[f+1]))
#             elif f==0:
#                 LC[f]=LC[f+1]
#             elif f==len(LC)-1:
#                 LC[f]=LC[f-1]
    if outlier==True:            
        LC=outlierr(LC,5,3)
        LC=outlierr(LC,5,3)
#     LC=outlierr(LC,5,3)
        
    z=avg
    oot_t0=np.empty([len(time0)/z])*np.nan
    oot_F0=np.empty([len(time0)/z])*np.nan
    k=0
    for i in range(0,len(oot_t0)):
        oot_t0[i]=np.nanmedian(time0[k:k+z])
        oot_F0[i]=np.nanmedian(LC[k:k+z])
        k+=z


    timein=timein
    timeeg=timeeg
    
   
     
    ### polynomial fit ###
    oot_F=np.append(oot_F0[np.where(oot_t0<timein)],oot_F0[np.where(oot_t0>timeeg)])
    oot_t=np.append(oot_t0[np.where(oot_t0<timein)],oot_t0[np.where(oot_t0>timeeg)])

    oot_t=oot_t[~np.isnan(oot_F)]
    oot_F=oot_F[~np.isnan(oot_F)]
    
    oot_t=oot_t[time_skip:len(oot_t)-time_trim]
    oot_F=oot_F[time_skip:len(oot_F)-time_trim]
   
    test=np.polyfit(oot_t,oot_F,order)
    tests=np.poly1d(test)
    out=tests(time0)
    out0=tests(oot_t0)
    
    new=(LC)/out
    
#         print '****************'
#         print np.min(X_arr),np.max(X_arr)
#         print X_arr
#         print '****************'
#         print np.min(LC_oott),np.max(LC_oott)
#         print LC_oott
#         print '****************'
#         print np.min(beta_arr),np.max(beta_arr)
#         print beta_arr

    #print np.nanmedian(err_p)
    err_p/=out
    print np.nanmedian(err_p)*10**6.

   
    ##################################
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(time0,LC,'.',markersize=9,color='darkgrey',alpha=0.3)
    ax[0].plot(oot_t,oot_F,'.',markersize=9.,color='dimgrey')
    ax[0].axvline(x=timein,color='darkgrey')
    ax[0].axvline(x=timeeg,color='darkgrey')
    #ax[0].plot(oot_t,oot_F,'.',markersize=4,color='white')
    ax[0].plot(time0,out,'-',color='black')
    #ax[0].set_ylim(0.97,1.06)
    ax[0].set_title('WHITE')
    ax[0].set_xlabel('Time,[hrs]')
    ax[0].set_ylabel('Relative Flux [hrs]')
    #ax[0].set_ylim(ybot,ytop)

    ### rmse estimate for LC fits ###
    ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
    Fa=(np.append(new[np.where(time0<timein)],new[np.where(time0>timeeg)]))

    ta=ta[~np.isnan(Fa)]
    Fa=Fa[~np.isnan(Fa)]

    RMSE_est=np.sqrt(np.nanmean(((Fa)-1.0)**2.))
    RMSE_est_orig=RMSE_est


    ax[1].plot(time0,new,'.',color='grey',alpha=0.3)
    ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color='dimgrey')
    ax[1].plot(time0,np.ones_like(time0),'-',color='black')
    ax[1].set_ylim(ybot,ytop)
    ax[1].set_title(str(np.round(RMSE_est,6)),fontsize=15)
    ax[1].set_xlabel('Time,[hrs]')
    ax[1].set_ylabel('Relative Flux [hrs]')
    plt.show()

        
   
    
    np.savez_compressed(SAVEPATH+'LCwhite_br.npz',data=new,time=time0,rmse=RMSE_est,
                            rmse_orig=RMSE_est_orig,polyfit=out,
                            err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0)

        
def blfit_binns(SAVEPATH,width,order,avg,olow,ohigh,ybot,ytop,timein,timeeg,
                corr,noise_white,spot,time_trim,time_skip,outlier):

    order=order
    low=olow
    high=ohigh

    ybot=ybot
    ytop=ytop
    
    z=avg

    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    n_exp=len(time0)

    if corr==True:
        LC_l=np.load(SAVEPATH+'LC_bins_'+str(int(width))+'_Corr.npz')['data']
        err_t=np.load(SAVEPATH+'LC_bins_'+str(int(width))+'_Corr.npz')['err_t']
        err_p=np.load(SAVEPATH+'LC_bins_'+str(int(width))+'_Corr.npz')['err_p']
    else:
        LC_l=np.load(SAVEPATH+'LC_bins_'+str(int(width))+'.npz')['data']
        err_t=np.load(SAVEPATH+'LC_bins_'+str(int(width))+'.npz')['err_t']
        err_p=np.load(SAVEPATH+'LC_bins_'+str(int(width))+'.npz')['err_p']
        
    new=np.empty([len(time0),LC_l.shape[1]])
    
    bin_arr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bins']
    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_centers']

    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-1]+2*width])
    print bin_arr
    
#     for t in range(0,len(time0)):
#         if t>len(time0)-time_trim:
#             LC_l[t,:]=np.nan
    
    if noise_white==True:
        if spot==True:
            lc_fitw=np.load(SAVEPATH+'/LightCurve_fits_spot.npz')['lightcurve_fit']
            white_residuals=np.load(SAVEPATH+'LightCurve_fits_spot.npz')['residuals']*10**-6
        else:
            lc_fitw=np.load(SAVEPATH+'LightCurve_fits_white.npz')['lightcurve_fit']
            white_residuals=np.load(SAVEPATH+'LightCurve_fits_white.npz')['residuals']*10**-6
        lc_dataw_corr=np.load(SAVEPATH+'LCwhite_br.npz')['data']#lc_fitw-white_residuals
        if corr==True:
            lc_dataw=np.load(SAVEPATH+'LCwhite_Corr.npz')['data']
        else:
            lc_dataw=np.load(SAVEPATH+'LCwhite.npz')['data']
        common_white=lc_dataw/lc_fitw
        #print lc_dataw
        #######################
        print '------------------- Common Mode Noise -------------------'
        fig,ax=plt.subplots(1,3,figsize=(15,4))
        ax[0].plot(time0,lc_dataw,'.',markersize=9,color='slateblue',alpha=0.7)
        ax[0].set_title('Raw White Curve',fontsize=15)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Relative Flux')
        #ax[0].set_ylim(ybot,ytop)

        ax[1].plot(time0,lc_dataw_corr,'.',markersize=9,color='slateblue',alpha=0.7)
        ax[1].plot(time0,lc_fitw,linewidth=1.0,color='grey')
        ax[1].set_title('Corrected White Curve',fontsize=15)
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Relative Flux')
        #ax[1].set_ylim(ybot,ytop)

        ax[2].plot(time0,common_white,markersize=9,color='slateblue',alpha=0.7)
        ax[2].set_title('COMMON MODE WHITE NOISE',fontsize=15)
        ax[2].set_xlabel('Time')
        #ax[2].set_ylim(ybot,ytop)

        plt.show()
        print ' --------------------------------------------------------- '
        #######################
        for b in range(0,LC_l.shape[1]):
            LC_l[:,b]/=common_white
            err_p[:,b]/=common_white
    else:
        common_white=np.zeros_like(LC_l[:,0])*np.nan
            
    
    
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])
    
    avgF=np.empty([len(time0)/z,len(bin_ctr)])*np.nan
    RMSE_est=np.empty([len(bin_ctr)])*np.nan
    RMSE_est_orig=np.empty([len(bin_ctr)])*np.nan
    
    polyfit=np.empty([len(time0),len(bin_ctr)])*np.nan
    
    print LC_l.shape
    for b in range(0,LC_l.shape[1]):
        print b
        if np.isnan(LC_l[int(n_exp/2),b])==True:# or np.isnan(LC_l[1,b])==True or np.isnan(LC_l[2,b])==True:
            print '---------', bin_ctr[b],'---------'
            continue
#         for f in range(0,LC_l.shape[0]):
#             if LC_l[f,b]<low or LC_l[f,b]>high or np.isfinite(LC_l[f,b])==False:
#                 if f>1 and f<LC_l.shape[0]-1:
#                     LC_l[f,b]=np.nanmedian(np.append(LC_l[f-1,b],LC_l[f+1,b]))
#                 if f==0:
#                     LC_l[f,b]=LC_l[f+1,b]
#                 if f==LC_l.shape[0]-1:
#                     LC_l[f,b]=LC_l[f-1,b]
                 
        if outlier==True:
            LC_l[:,b]=outlierr(LC_l[:,b],5,3)
            LC_l[:,b]=outlierr(LC_l[:,b],5,3)
            #LC_l[:,b]=outlierr(LC_l[:,b],5,3)
     
        LCb=LC_l[:,b]
        
    
        oot_t0=np.empty([len(time0)/z])*np.nan
        oot_F0=np.empty([len(time0)/z])*np.nan
        k=0
        for i in range(0,len(oot_t0)):
            oot_t0[i]=np.nanmedian(time0[k:k+z])
            oot_F0[i]=np.nanmedian(LCb[k:k+z])
            k+=z
    
        oot_F=np.append(oot_F0[np.where(oot_t0<timein)],oot_F0[np.where(oot_t0>timeeg)])
        oot_t=np.append(oot_t0[np.where(oot_t0<timein)],oot_t0[np.where(oot_t0>timeeg)])

        oot_t=oot_t[~np.isnan(oot_F)]
        oot_F=oot_F[~np.isnan(oot_F)]
    
        oot_t=oot_t[time_skip:len(oot_t)-time_trim]
        oot_F=oot_F[time_skip:len(oot_F)-time_trim]
        
    
        test=np.polyfit(oot_t,oot_F,order)
        tests=np.poly1d(test)
        out=tests(time0)
        out0=tests(oot_t0)

        new[:,b]=(LCb)/out
    
        avgF[:,b]=oot_F0/out0
        
        polyfit[:,b]=out
        
        #print np.nanmedian(err_p[:,b])
        err_p[:,b]/=polyfit[:,b]
        print np.nanmedian(err_p[:,b])*10**6.
        


        if noise_white==True:
            fig,ax=plt.subplots(1,3,figsize=(15,4))
            ax[0].plot(time0,LCb*common_white,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.8)
            #ax[0].plot(oot_t,oot_F,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
            #ax[0].plot(time0,out,'-',color='black')
            #ax[0].set_ylim(ybot-0.005,ytop+0.005)
            ax[0].set_title(str(int(bin_ctr[b])))
            ax[0].set_xlabel('Time,[hrs]')
            ax[0].set_ylabel('Relative Flux [hrs]')
            
            ax[1].plot(time0,LCb,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.35)
            ax[1].plot(oot_t,oot_F,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[1].plot(time0,out,'-',color='black')
            #ax[0].set_ylim(0.97,np.nanmax(LCb))
            ax[1].set_title(str(int(bin_ctr[b])))
            ax[1].set_xlabel('Time,[hrs]')
            ax[1].set_ylabel('Relative Flux [hrs]')
            #ax[1].set_ylim(ybot,ytop)
           
            
            ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
            Fa=(np.append(new[np.where(time0<timein),b],new[np.where(time0>timeeg),b]))
        
            ta=ta[~np.isnan(Fa)]
            Fa=Fa[~np.isnan(Fa)]

            RMSE_est[b]=np.sqrt(np.nanmean(((Fa)-1.0)**2.))
            RMSE_est_orig=RMSE_est
            noise_model_fit=np.zeros_like(new)

            ax[2].plot(time0,new[:,b],'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.35)
            ax[2].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[2].plot(time0,out/out,'-',color='black')
            ax[2].set_title(str(np.round(RMSE_est[b],6)),fontsize=15)
            ax[2].set_ylim(ybot,ytop)
            ax[2].set_xlabel('Time,[hrs]')
            ax[2].set_ylabel('Relative Flux [hrs]')
            plt.show()
        else:
            fig,ax=plt.subplots(1,2,figsize=(10,4))
            ax[0].plot(time0,LCb,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.35)
            ax[0].plot(oot_t,oot_F,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[0].plot(time0,out,'-',color='black')
            #ax[0].set_ylim(0.97,np.nanmax(LCb))
            ax[0].set_title(str(int(bin_ctr[b])))
            ax[0].set_xlabel('Time,[hrs]')
            ax[0].set_ylabel('Relative Flux [hrs]')
            #ax[0].set_ylim(ybot,ytop)
           
            
            ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
            Fa=(np.append(new[np.where(time0<timein),b],new[np.where(time0>timeeg),b]))
        
            ta=ta[~np.isnan(Fa)]
            Fa=Fa[~np.isnan(Fa)]

            RMSE_est[b]=np.sqrt(np.nanmean(((Fa)-1.0)**2.))
            RMSE_est_orig=RMSE_est
            noise_model_fit=np.zeros_like(new)

            ax[1].plot(time0,new[:,b],'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.35)
            ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[1].plot(time0,out/out,'-',color='black')
            ax[1].set_title(str(np.round(RMSE_est[b],6)),fontsize=15)
            ax[1].set_ylim(ybot,ytop)
            ax[1].set_xlabel('Time,[hrs]')
            ax[1].set_ylabel('Relative Flux [hrs]')
            plt.show()
            

    np.savez_compressed(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz',
                            data=new,cw=common_white,time=time0,rmse=RMSE_est,
                            rmse_orig=RMSE_est_orig,bin_ctr=bin_ctr,polyfit=polyfit,
                            err_t=err_t,err_p=err_p,avt=oot_t0,avf=avgF)