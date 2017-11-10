import scipy

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


from setup import *

def blfit_white(SAVEPATH,order,avg,olow,ohigh,ybot,ytop,timein,timeeg,corr):

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

    
    for f in range(0,len(LC)):
        if LC[f]<low or LC[f]>high:
            LC[f]=np.nan

        
    z=avg
    oot_t0=np.empty([len(time0)/z])
    oot_F0=np.empty([len(time0)/z])
    k=0
    for i in range(0,len(oot_t0)):
        oot_t0[i]=np.nanmedian(time0[k:k+z])
        oot_F0[i]=np.nanmedian(LC[k:k+z])
        k+=z


    timein=timein
    timeeg=timeeg
    
    oot_F=np.append(oot_F0[np.where(oot_t0<timein)],oot_F0[np.where(oot_t0>timeeg)])
    oot_t=np.append(oot_t0[np.where(oot_t0<timein)],oot_t0[np.where(oot_t0>timeeg)])

    oot_t=oot_t[~np.isnan(oot_F)]
    oot_F=oot_F[~np.isnan(oot_F)]
    
    oot_t=oot_t[:-1]
    oot_F=oot_F[:-1]
   

    test=np.polyfit(oot_t,oot_F,order)
    tests=np.poly1d(test)
    out=tests(time0)
    out0=tests(oot_t0)

    new=(LC)/out

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

    ax[1].plot(time0,new,'.',color='grey',alpha=0.3)
    ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color='dimgrey')
    ax[1].plot(time0,out/out,'-',color='black')
    ax[1].set_ylim(ybot,ytop)
    ax[1].set_title('WHITE')
    ax[1].set_xlabel('Time,[hrs]')
    ax[1].set_ylabel('Relative Flux [hrs]')
    plt.show()

    if corr==True:
        np.savez_compressed(SAVEPATH+'LCwhite_br_Corr.npz',data=new,time=time0,err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0)
    else:
        np.savez_compressed(SAVEPATH+'LCwhite_br.npz',data=new,time=time0,err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0)

        
def blfit_binns(SAVEPATH,width,order,avg,olow,ohigh,ybot,ytop,timein,timeeg,corr):

    order=order
    low=olow
    high=ohigh

    ybot=ybot
    ytop=ytop

    time0=np.load(SAVEPATH+'Obs_times.npz')['times']

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

    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-2]+width])
    print bin_arr
    
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])

    for b in range(0,LC_l.shape[1]):
        if np.isnan(LC_l[2,b])==True:
            continue
        for f in range(0,LC_l.shape[0]):
            if LC_l[f,b]<low or LC_l[f,b]>high:
                LC_l[f,b]=np.nan
     
        LCb=LC_l[:,b]
    
        z=avg
        oot_t0=np.empty([len(time0)/z])
        oot_F0=np.empty([len(time0)/z])
        k=0
        for i in range(0,len(oot_t0)):
            oot_t0[i]=np.nanmedian(time0[k:k+z])
            oot_F0[i]=np.nanmedian(LCb[k:k+z])
            k+=z
    
        oot_F=np.append(oot_F0[np.where(oot_t0<timein)],oot_F0[np.where(oot_t0>timeeg)])
        oot_t=np.append(oot_t0[np.where(oot_t0<timein)],oot_t0[np.where(oot_t0>timeeg)])

        oot_t=oot_t[~np.isnan(oot_F)]
        oot_F=oot_F[~np.isnan(oot_F)]
    
        oot_t=oot_t[:-1]
        oot_F=oot_F[:-1]
    
        test=np.polyfit(oot_t,oot_F,order)
        tests=np.poly1d(test)
        out=tests(time0)
        out0=tests(oot_t0)

        new[:,b]=(LCb)/out
    
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].plot(time0,LCb,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.2)
        ax[0].plot(oot_t,oot_F,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
        ax[0].plot(time0,out,'-',color='black')
        #ax[0].set_ylim(0.97,np.nanmax(LCb))
        ax[0].set_title(str(int(bin_ctr[b])))
        ax[0].set_xlabel('Time,[hrs]')
        ax[0].set_ylabel('Relative Flux [hrs]')

        ax[1].plot(time0,new[:,b],'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.2)
        ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color=scal_m.to_rgba(bin_ctr[b]+width))
        ax[1].plot(time0,out/out,'-',color='black')
        ax[1].set_ylim(ybot,ytop)
        ax[1].set_title(str(int(bin_ctr[b])))
        ax[1].set_xlabel('Time,[hrs]')
        ax[1].set_ylabel('Relative Flux [hrs]')
        plt.show()

    if corr==True:
        np.savez_compressed(SAVEPATH+'LC_bins_br_'+str(int(width))+'_Corr.npz',data=new,time=time0,bin_ctr=bin_ctr,err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0)
    else:
        np.savez_compressed(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz',data=new,time=time0,bin_ctr=bin_ctr,err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0)