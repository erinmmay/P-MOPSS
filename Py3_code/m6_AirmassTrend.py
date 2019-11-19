from scipy.optimize import curve_fit
from numpy.linalg import inv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def airmass_func(t,k):
    d0=data[obj,int(n_exp/2.),b]
    z0=z[int(n_exp/2.)]
    return d0*10**(-1.0*k*(z-z0)/2.5)

def Airmass_White(SAVEPATH,width,timein,timeeg,ybot,ytop,obj_skip):

#     k_arr=airmass_fit(SAVEPATH,obj_skip)
#     print k_arr
    global b
    global data
    global time0
    global z
    global obj
    global n_exp
    
    b=0
    
    data=np.load(SAVEPATH+'Binned_Data_White.npz')['bin_counts']
    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    
    inds=np.isfinite(data[0,:,0])
    time0=time0[inds]
    data=data[:,inds,:]
    
    n_exp=len(time0)
    n_obj=data.shape[0]
   
    ##
    z=(np.load(SAVEPATH+'HeaderData.npz')['airmass'])
    z=z[inds]
    ##
    k_white=np.empty([n_obj])*np.nan
    
    for o in range(0,n_obj):
        if o in obj_skip:# or o==0:
            continue
        obj=o
        
        k_white[obj],F_cov=curve_fit(airmass_func,time0,data[obj,:,0],p0=[0])
        ##
        fig,ax=plt.subplots(3,1,figsize=(15,8))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].plot(time0,data[obj,:,0],'.',markersize=15,markerfacecolor='tomato',markeredgecolor='black')
        ax[1].plot(time0,z,'-',color='tomato',linewidth=2.5)
        ax[1].set_ylim(max(z)+0.1,min(z)-0.1)
        ax[2].plot(time0,data[obj,:,0],'.',markersize=15,markerfacecolor='tomato',markeredgecolor='black')
        ax[2].plot(time0,airmass_func(time0,k_white[obj]),'-',color='grey',linewidth=2.5)
        plt.figtext(0.17,0.7,str(obj),fontsize=35,color='black')
        plt.show(block=False)
    ###############
    k=np.nanmedian(k_white)
    print(k)

    ## remove airmass extinction ##
    data_rmZ=np.empty([data.shape[0],data.shape[1]])*np.nan

    fig,ax=plt.subplots(data.shape[0],1,figsize=(15,20))
    fig.subplots_adjust(wspace=0, hspace=0)
    for o in range(0,data.shape[0]):
        if o in obj_skip:
            continue
        obj=o
        data_rmZ[obj,:]=data[obj,:,0]/airmass_func(time0,k)

        ax[obj].plot(time0,data_rmZ[obj,:],'.',markersize=12,markerfacecolor='grey',markeredgecolor='black')
        ax[obj].axhline(y=1.0,color='black',linewidth=2.5)
        ax[obj].set_ylim(ybot,ytop)
    plt.show(block=False)


    print('---------------------------')
    np.savez_compressed(SAVEPATH+'AirmassRemove_White.npz',data=data_rmZ,k=k_white)
    
def Airmass_Binns(SAVEPATH,width,timein,timeeg,ybot,ytop,obj_skip):
    
    bin_arr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bins']
    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_centers']

    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-1]+2*width])
    print(bin_arr)
    
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])
    
#     k_arr=airmass_fit(SAVEPATH,obj_skip)
#     print k_arr

    global b
    global data
    global time0
    global z
    global obj
    global n_exp

    data=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_counts']
    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    
        
    inds=np.isfinite(data[0,:,0])
    time0=time0[inds]
    data=data[:,inds,:]
    
    n_exp=len(time0)
    n_obj=data.shape[0]
    n_bins=data.shape[2]

    ##
    z=(np.load(SAVEPATH+'HeaderData.npz')['airmass'])
    z=z[inds]
    ##
    k_binns=np.empty([n_obj,n_bins])*np.nan
    for o in range(0,n_obj):
        if o in obj_skip:# or o==0:
            continue
        obj=o
        for b in range(0,n_bins):
            k_binns[obj,b],F_cov=curve_fit(airmass_func,time0,data[obj,:,b],p0=[0])
            ##
#             fig,ax=plt.subplots(3,1,figsize=(15,8))
#             fig.subplots_adjust(wspace=0, hspace=0)
#             ax[0].plot(time0,data[obj,:,0],'.',markersize=15,markerfacecolor='tomato',markeredgecolor='black')
#             ax[1].plot(time0,z,'-',color='tomato',linewidth=2.5)
#             ax[1].set_ylim(max(z)+0.1,min(z)-0.1)
#             ax[2].plot(time0,data[obj,:,0],'.',markersize=15,markerfacecolor='tomato',markeredgecolor='black')
#             ax[2].plot(time0,airmass_func(time0,k_binns[obj,b]),'-',color='grey',linewidth=2.5)
#             plt.figtext(0.17,0.7,str(obj),fontsize=35,color='black')
#             plt.show(block=False)
    ###############
    k_fit=np.nanmedian(k_binns,axis=0)
    print(k_fit)

    ## remove airmass extinction ##
    data_rmZ=np.empty(data.shape)*np.nan

    for o in range(0,n_obj):
        if o in obj_skip:
            continue
        obj=o
        fig,ax=plt.subplots(data.shape[2],1,figsize=(15,3*n_bins))
        fig.subplots_adjust(wspace=0, hspace=0)
        
        for b in range(0,n_bins):
            data_rmZ[obj,:,b]=data[obj,:,b]/airmass_func(time0,k_fit[b])

            ax[b].plot(time0,data_rmZ[obj,:,b],'.',markersize=12,
                       markerfacecolor=scal_m.to_rgba(bin_ctr[b]),markeredgecolor='black')
            ax[b].axhline(y=1.0,color='grey',linewidth=2.5)
            ax[b].set_ylim(ybot,ytop)
            plt.figtext(0.2,0.9,str(obj),fontsize=35,color='black')
            
        plt.show(block=False)


    print('---------------------------')
    np.savez_compressed(SAVEPATH+'AirmassRemove_'+str(int(width))+'.npz',
                        data=data_rmZ,k=k_fit)