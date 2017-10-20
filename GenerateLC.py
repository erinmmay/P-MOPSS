import numpy as np
import matplotlib.pyplot as plt

from setup import *

def LCwhite(DATAFILE,TIMEFILE,SAVEPATH,check,cals,ts,ll,ul):
    Data=np.load(DATAFILE)['bin_counts']
    time0=np.load(TIMEFILE)['times']
    
    Cal_data=np.empty([Data.shape[0]])*0.0
    for c in range(0,len(cals)):
        Cal_data[:]=np.nansum(Cal_data[:],Data[:,0,c])
    Cal_data=Cal_data/np.nanmean(Cal_data)
    
    LC_data=Data[:,0,0]/Cal_data
    LC_data/=(LC_data[:ts])
    
    CS_data=Data[:,0,check]/Cal_data
    CS_data/=np.nanmean(CS_data[:ts])
    
    for t in range(0,len(LC_data)):
        if ll>0 and LC_data[t]<ll:
            LC_data[t]=np.nan
            CS_data[t]=np.nan
        if ul>0 and LC_data[t]>ul:
            LC_data[t]=np.nan
            CS_data[t]=np.nan
    
    fig,ax=plt.subplots(1,2,figsize=(10,4))

    ax[0].plot(time0,LC_data,'.',color='grey')
    ax[0].set_ylim(np.nanmin(LC_data)-0.01,np.nanmax(LC_data)+0.01)
    ax[0].set_title('WHITE LIGHT')
    ax[0].set_xlabel('Time,[days]')
    ax[0].set_ylabel('Relative Flux')
    
    ax[1].plot(time0,CS_data,'.',color='grey')
    ax[1].set_ylim(np.nanmin(LC_data)-0.01,np.nanmax(LC_data)+0.01)
    ax[1].set_title('Check Star - WHITE LIGHT')
    ax[1].set_xlabel('Time,[days]')
    #ax[1].set_ylabel('Relative Flux')
  
    plt.savefig(SAVEPATH+'LCwhite.png')
    plt.show(block=False)
    plt.pause(2.0)
    
    np.savez(SAVEPATH+'LCwhite.npz',data=LC_data,cs=CS_data,cals=Cal_data,time=time0)
 
    
def LClam(DATAFILE,TIMEFILE,SAVEPATH,check,cals,ts,ll,ul):
    
    
    Data=np.load(DATAFILE)['bin_counts']
    
    #bin_arr=np.load(DATAFILE)['bins']
    bin_ctr=np.load(DATAFILE)['bin_centers']

    time0=np.load(TIMEFILE)['times']
    
    ################################
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_ctr),vmax=np.max(bin_ctr))
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])
    ################################
    
    Cal_data=np.empty([Data.shape[0],len(bin_ctr)])*0.0
    LC_data=np.empty([Data.shape[0],len(bin_ctr)])*0.0
    CS_data=np.empty([Data.shape[0],len(bin_ctr)])*0.0
    
    for b in range(0,len(bin_ctr)):
        for c in range(0,len(cals)):
            Cal_data[:,b]=np.nansum(Cal_data[:],Data[:,b,c])
        Cal_data[:,b]=Cal_data[:,b]/np.nanmean(Cal_data[:,b])
    
        LC_data[:,b]=Data[:,b,0]/Cal_data[:,b]
        LC_data[:,b]/=(LC_data[:ts,b])
    
        CS_data[:,b]=Data[:,b,check]/Cal_data[:,b]
        CS_data[:,b]/=np.nanmean(CS_data[:ts,b])
    
        for t in range(0,len(LC_data[:,b])):
            if ll>0 and LC_data[t,b]<ll:
                LC_data[t,b]=np.nan
                CS_data[t,b]=np.nan
            if ul>0 and LC_data[t,b]>ul:
                LC_data[t,b]=np.nan
                CS_data[t,b]=np.nan
    
        fig,ax=plt.subplots(1,2,figsize=(10,4))

        ax[0].plot(time0,LC_data[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        ax[0].set_ylim(np.nanmin(LC_data[:,b])-0.01,np.nanmax(LC_data[:,b])+0.01)
        ax[0].set_title(str(int(bin_ctr[b])))
        ax[0].set_xlabel('Time,[days]')
        ax[0].set_ylabel('Relative Flux')
    
        ax[1].plot(time0,CS_data[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        ax[1].set_ylim(np.nanmin(LC_data[:,b])-0.01,np.nanmax(LC_data[:,b])+0.01)
        ax[1].set_title('Check Star - ',str(int(bin_ctr[b])))
        ax[1].set_xlabel('Time,[days]')
        
        plt.savefig(SAVEPATH+'LCbin_'+str(int(bin_ctr[b]))+'.png')
        plt.show(block=False)
        plt.pause(2.0)
        
    np.savez(SAVEPATH+'LC_bins.npz',data=LC_data,cs=CS_data,cals=Cal_data,time=time0,bin_ctr=bin_ctr)