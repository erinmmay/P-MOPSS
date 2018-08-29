from scipy.optimize import curve_fit
from numpy.linalg import inv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def airmass_func(t,k):
    d0=data[obj,int(n_exp/2.),b]
    z0=z[int(n_exp/2.)]
    return d0*10**(-1.0*k*(z-z0)/2.5)

def NoiseRun_White(SAVEPATH,width,timein,timeeg,ybot,ytop,obj_skip):

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
    n_exp=len(time0)
    n_obj=data.shape[0]

    ##
    z=(np.load(SAVEPATH+'HeaderData.npz')['airmass'])
    ##
    k_white=np.empty([n_obj])*np.nan
    
    for o in range(0,n_obj):
        if o in obj_skip or o==0:
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
    print k

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


    print '---------------------------'

    ###########
    data_clean=np.empty(data_rmZ.shape)*np.nan

    model_inputs=np.load(SAVEPATH+'NoiseModel_Inputs_'+str(int(width))+'.npz')
    noise_model_fit=np.empty([n_obj,n_exp])*np.nan
    
    for o in range(0,n_obj):
        if o in obj_skip:
            continue
        dataf=data_rmZ[o,:]

        X_loc=((model_inputs['white_x'])[o,:]-(model_inputs['white_x'])[o,0]).reshape(-1,1)
        Y_loc=((model_inputs['yshift'])[o,:]).reshape(-1,1)#(model_inputs['yshift']).reshape(-1,1)
        bg_ct=((model_inputs['white_bg'])[o,:]).reshape(-1,1)
        fwhm_=(np.load(SAVEPATH+'FlattenedSpectra.npz')['gaus_params'])[o,:,:,2]#'fwhm_ar'])[0,:].reshape(-1,1)
        fwhm_=(2.*np.sqrt(2.*np.log(2.))*np.nanmedian(fwhm_,axis=1)).reshape(-1,1)
        plt.show(block=False)
        ones=(np.ones(n_exp)).reshape(-1,1)

        time_fl=(np.copy(time0)).reshape(-1,1)  #Full Time
        model_stack_full=np.hstack((ones,fwhm_,X_loc,Y_loc,bg_ct))  #Full Matrix

        out_index=np.array([],dtype=int)
        for t in range(0,len(time0)):
            if time0[t]<timein or time0[t]>timeeg:
                out_index=np.append(out_index,int(t))
        n_out=len(out_index)  

        model_stack_oott=model_stack_full[out_index,:]

        ###
        fig,ax=plt.subplots(4,1,figsize=(15,10))
        fig.subplots_adjust(wspace=0, hspace=0.0)
        ax[0].plot(time_fl,X_loc,linewidth=4.0,color='slateblue',alpha=0.7)
        ax[0].set_title('X_shift')
        ax[1].plot(time_fl,Y_loc,linewidth=4.0,color='slateblue',alpha=0.7)
        ax[1].set_title('Y_shift')
        ax[2].plot(time_fl,bg_ct,linewidth=4.0,color='slateblue',alpha=0.7)
        ax[2].set_title('Background Counts')
        ax[3].plot(time_fl,fwhm_,linewidth=4.0,color='slateblue',alpha=0.7)
        ax[3].set_title('FWHM')
        plt.show()
        ###

        if o==0:  
            X_arr=model_stack_oott
            LC_oott=np.append(dataf[np.where(time0<timein)],dataf[np.where(time0>timeeg)])
            beta_arr=np.dot(np.dot(inv(np.dot(X_arr.T,X_arr)),X_arr.T),LC_oott)
        else:
            X_arr=model_stack_full
            beta_arr=np.dot(np.dot(inv(np.dot(X_arr.T,X_arr)),X_arr.T),dataf)

        noise_model_fit[o,:]=np.dot(model_stack_full,beta_arr)
        data_clean[o,:]=dataf/noise_model_fit[o,:]

        fig,ax=plt.subplots(1,2,figsize=(10,4))
        fig.subplots_adjust(wspace=0, hspace=0.0)
        ax[0].plot(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]),
                   np.append(dataf[np.where(time0<timein)],dataf[np.where(time0>timeeg)]),
                   '.',color='grey',markersize=9)
        ax[0].plot(time0,dataf,'.',color='grey',markersize=9,alpha=0.5)
        ax[0].plot(time0,noise_model_fit[o,:],color='black',linewidth=1.5)
        ax[0].set_ylim(ybot,ytop)
        ax[1].plot(time0,data_clean[o,:],'.',color='grey',markersize=9)
        ax[1].axhline(y=1.0,color='black',linewidth=1.5)
        ax[1].set_ylim(ybot,ytop)
        plt.figtext(0.2,0.8,str(o),fontsize=35,color='black')
        plt.show(block=False)

    np.savez_compressed(SAVEPATH+'NoiseModel_FitResults_White.npz',data=data_clean,k=k,noise_model=noise_model_fit)

    ################################
def NoiseRun_Binns(SAVEPATH,width,timein,timeeg,ybot,ytop,obj_skip):
    
    bin_arr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bins']
    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_centers']

    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-1]+2*width])
    print bin_arr
    
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
    n_exp=len(time0)
    n_obj=data.shape[0]
    n_bins=data.shape[2]

    ##
    z=(np.load(SAVEPATH+'HeaderData.npz')['airmass'])
    ##
    k_binns=np.empty([n_obj,n_bins])*np.nan
    for o in range(0,n_obj):
        if o in obj_skip or o==0:
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
    print k_fit

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


    print '---------------------------'

    ###########
    data_clean=np.empty(data_rmZ.shape)*np.nan

    model_inputs=np.load(SAVEPATH+'NoiseModel_Inputs_'+str(int(width))+'.npz')
    noise_model_save=np.empty([n_obj,n_bins,n_exp])*np.nan
    for o in range(0,n_obj):
        if o in obj_skip:
            continue
        
        for b in range(0,n_bins):
            dataf=data_rmZ[o,:,b]
            if dataf[0]==np.nan:
                print bin_ctr[b], 'A -> SKIPPED BIN. Is this in chip gap??'
                continue

            X_loc=((model_inputs['binned_x'])[o,:,b]-(model_inputs['binned_x'])[o,0,b]).reshape(-1,1)
            Y_loc=((model_inputs['yshift'])[o,:]).reshape(-1,1)#(model_inputs['yshift']).reshape(-1,1)
            bg_ct=((model_inputs['binned_bg'])[o,:,b]).reshape(-1,1)
            fwhm_=(np.load(SAVEPATH+'FlattenedSpectra.npz')['gaus_params'])[o,:,:,2]#'fwhm_ar'])[0,:].reshape(-1,1)
            fwhm_=(2.*np.sqrt(2.*np.log(2.))*np.nanmedian(fwhm_,axis=1)).reshape(-1,1)
            ones=(np.ones(n_exp)).reshape(-1,1)

            time_fl=(np.copy(time0)).reshape(-1,1)  #Full Time
            model_stack_full=np.hstack((ones,fwhm_,X_loc,Y_loc,bg_ct))  #Full Matrix

            out_index=np.array([],dtype=int)
            for t in range(0,len(time0)):
                if time0[t]<timein or time0[t]>timeeg:
                    out_index=np.append(out_index,int(t))
            n_out=len(out_index)  

            model_stack_oott=model_stack_full[out_index,:]

            ###
#             fig,ax=plt.subplots(4,1,figsize=(15,10))
#             fig.subplots_adjust(wspace=0, hspace=0.0)
#             ax[0].plot(time_fl,X_loc,linewidth=4.0,color='slateblue',alpha=0.7)
#             ax[0].set_title('X_shift')
#             ax[1].plot(time_fl,Y_loc,linewidth=4.0,color='slateblue',alpha=0.7)
#             ax[1].set_title('Y_shift')
#             ax[2].plot(time_fl,bg_ct,linewidth=4.0,color='slateblue',alpha=0.7)
#             ax[2].set_title('Background Counts')
#             ax[3].plot(time_fl,fwhm_,linewidth=4.0,color='slateblue',alpha=0.7)
#             ax[3].set_title('FWHM')
#             plt.show()
            ###

            if o==0:  
                X_arr=model_stack_oott
                LC_oott=np.append(dataf[np.where(time0<timein)],dataf[np.where(time0>timeeg)])
                beta_arr=np.dot(np.dot(inv(np.dot(X_arr.T,X_arr)),X_arr.T),LC_oott)
            else:
                X_arr=model_stack_full
                beta_arr=np.dot(np.dot(inv(np.dot(X_arr.T,X_arr)),X_arr.T),dataf)

            noise_model_fit=np.dot(model_stack_full,beta_arr)
            noise_model_save[o,b,:]=noise_model_fit
            data_clean[o,:,b]=dataf/noise_model_fit

            fig,ax=plt.subplots(1,2,figsize=(10,4))
            fig.subplots_adjust(wspace=0, hspace=0.0)
            ax[0].plot(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]),
                   np.append(dataf[np.where(time0<timein)],dataf[np.where(time0>timeeg)]),
                       '.',color=scal_m.to_rgba(bin_ctr[b]),markersize=9)
            ax[0].plot(time0,dataf,'.',color=scal_m.to_rgba(bin_ctr[b]),markersize=9,alpha=0.5)
            ax[0].plot(time0,noise_model_fit,color='grey',linewidth=1.5)
            ax[0].set_ylim(ybot,ytop)
            ax[1].plot(time0,data_clean[o,:,b],'.',color=scal_m.to_rgba(bin_ctr[b]),markersize=9)
            ax[1].axhline(y=1.0,color='grey',linewidth=1.5)
            ax[1].set_ylim(ybot,ytop)
            plt.figtext(0.2,0.8,str(o),fontsize=35,color='black')
            plt.show(block=False)

    np.savez_compressed(SAVEPATH+'NoiseModel_FitResults_'+str(int(width))+'.npz',
                        data=data_clean,k=k_fit,noise_model=noise_model_save)
