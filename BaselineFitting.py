import scipy

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from numpy.linalg import inv


from setup import *

def blfit_white(SAVEPATH,order,avg,olow,ohigh,ybot,ytop,timein,timeeg,corr,noise_red):

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
        
        
    if noise_red==True:
        n_exp=len(time0)
        model_inputs=np.load(SAVEPATH+'NoiseModel_Inputs_200.npz')
        X_loc=(model_inputs['white_x']-(model_inputs['white_x'])[0]).reshape(-1,1)
        X_sq=(X_loc*X_loc)
        Y_loc=(model_inputs['yshift']).reshape(-1,1)
        Y_sq=(Y_loc*Y_loc)
        
        XY=(X_loc*Y_loc)
        
        bg=(model_inputs['white_bg']).reshape(-1,1)
        
        airmass=(np.load(SAVEPATH+'HeaderData.npz')['airmass']).reshape(-1,1)
        median_fwhm=(np.load(SAVEPATH+'FlattenedSpectra.npz')['fwhm_av'])[0,:].reshape(-1,1)
        ones=(np.ones(n_exp)).reshape(-1,1)
            
        time_fl=(np.copy(time0)).reshape(-1,1)
        
       
            
        model_stack_full=np.hstack((ones,airmass,median_fwhm,X_loc,Y_loc,bg,X_sq,Y_sq,XY))
        
        out_index=np.array([],dtype=int)
        for t in range(0,len(time0)):
            if time0[t]<timein or time0[t]>timeeg:
                out_index=np.append(out_index,int(t))
        n_out=len(out_index)  
        
        #model_stack_oott=np.ones([n_out,model_stack_full.shape[1]])
        model_stack_oott=model_stack_full[out_index,:]
        
        #ind=0
        #while ind<n_exp:
        #    for i in out_index:
        #        model_stack_oott[ind,:]=model_stack_full[out_index,:]
        #    if time0[ind]<timein or time0[ind]>timeeg:
        #        model_stack_oott[ind,:]=model_stack_full[ind,:]
        #    ind+=1
        
        #model_stack_oott=np.hstack((ones[tt],airmass[tt],median_fwhm[tt],X_loc[tt],Y_loc[tt],bg[tt]))
#         fig,ax=plt.subplots(3,3,figsize=(15,12))
        
#         ax[0,0].plot(time_fl,X_loc,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[0,0].set_title('X_shift')
        
#         ax[0,1].plot(time_fl,Y_loc,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[0,1].set_title('Y_shift')
        
#         ax[0,2].plot(time_fl,bg,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[0,2].set_title('Background Counts')
#         ###
#         ax[1,0].plot(time_fl,X_sq,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[1,0].set_title('X^2')
        
#         ax[1,1].plot(time_fl,Y_sq,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[1,1].set_title('Y^2')
        
#         ax[1,2].plot(time_fl,XY,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[1,2].set_title('XY')
#         ###
#         ax[2,0].plot(time_fl,airmass,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[2,0].set_title('Z')
        
#         ax[2,1].plot(time_fl,median_fwhm,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[2,1].set_title('FWHM')
       
#         ax[2,2].plot(time_fl,ones,linewidth=4.0,color='slateblue',alpha=0.7)
#         ax[2,2].set_title('1')
        
#         plt.show()
        
        
    
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
    
   
     
    ### polynomial fit ###
    oot_F=np.append(oot_F0[np.where(oot_t0<timein)],oot_F0[np.where(oot_t0>timeeg)])
    oot_t=np.append(oot_t0[np.where(oot_t0<timein)],oot_t0[np.where(oot_t0>timeeg)])

    oot_t=oot_t[~np.isnan(oot_F)]
    oot_F=oot_F[~np.isnan(oot_F)]
    
    oot_t=oot_t[:]
    oot_F=oot_F[:]
   
    test=np.polyfit(oot_t,oot_F,order)
    tests=np.poly1d(test)
    out=tests(time0)
    out0=tests(oot_t0)
    
    new=(LC)/out
    
    if noise_red==True:
        X_arr=model_stack_oott
        LC_oott=np.append(new[np.where(time0<timein)],new[np.where(time0>timeeg)])
        beta_arr=np.dot(np.dot(inv(np.dot(X_arr.T,X_arr)),X_arr.T),LC_oott)
        
        noise_model_fit=np.dot(model_stack_full,beta_arr)

   
    ##################################
    if noise_red==True:
        fig,ax=plt.subplots(1,3,figsize=(15,4))
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
        
        ax[1].plot(time0,new,'.',markersize=9,color='darkgrey',alpha=0.3)
        ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=9.,color='dimgrey')
        ax[1].axvline(x=timein,color='darkgrey')
        ax[1].axvline(x=timeeg,color='darkgrey')
        #ax[0].plot(oot_t,oot_F,'.',markersize=4,color='white')
        ax[1].plot(time0,noise_model_fit,'-',color='black')
        #ax[0].set_ylim(0.97,1.06)
        ax[1].set_title('WHITE')
        ax[1].set_xlabel('Time,[hrs]')
        ax[1].set_ylabel('Relative Flux [hrs]')
        
        new=new/noise_model_fit
        
        ### rmse estimate for LC fits ###
        ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
        Fa=(np.append(new[np.where(time0<timein)],new[np.where(time0>timeeg)]))
    
        ta=ta[~np.isnan(Fa)]
        Fa=Fa[~np.isnan(Fa)]
    
        RMSE_est=np.sqrt(np.nanmean(((Fa)-1.0)**2.))

        ax[2].plot(time0,new,'.',markersize=9,color='grey',alpha=0.5)
        #ax[2].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color='dimgrey')
        ax[2].plot(time0,np.ones_like(time0),'-',color='black')
        ax[2].set_ylim(ybot,ytop)
        ax[2].set_title(str(np.round(RMSE_est,6)),fontsize=15)
        ax[2].set_xlabel('Time,[hrs]')
        ax[2].set_ylabel('Relative Flux [hrs]')
        plt.show()
    else:
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
        
        ### rmse estimate for LC fits ###
        ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
        Fa=(np.append(new[np.where(time0<timein)],new[np.where(time0>timeeg)]))
    
        ta=ta[~np.isnan(Fa)]
        Fa=Fa[~np.isnan(Fa)]
    
        RMSE_est=np.sqrt(np.nanmean(((Fa)-1.0)**2.))

        ax[1].plot(time0,new,'.',color='grey',alpha=0.3)
        ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color='dimgrey')
        ax[1].plot(time0,np.ones_like(time0),'-',color='black')
        ax[1].set_ylim(ybot,ytop)
        ax[1].set_title(str(np.round(RMSE_est,6)),fontsize=15)
        ax[1].set_xlabel('Time,[hrs]')
        ax[1].set_ylabel('Relative Flux [hrs]')
        plt.show()

        
   
    
    if corr==True:
        np.savez_compressed(SAVEPATH+'LCwhite_br_Corr.npz',data=new,time=time0,rmse=RMSE_est,
                            err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0,beta=beta_arr)
    else:
        np.savez_compressed(SAVEPATH+'LCwhite_br.npz',data=new,time=time0,rmse=RMSE_est,
                            err_t=err_t,err_p=err_p,avt=oot_t0,avf=oot_F0/out0,beta=beta_arr)

        
def blfit_binns(SAVEPATH,width,order,avg,olow,ohigh,ybot,ytop,timein,timeeg,corr,white,noise_red):

    order=order
    low=olow
    high=ohigh

    ybot=ybot
    ytop=ytop
    
    z=avg

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

    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-1]+2*width])
    print bin_arr
    
    if white==True:
        white_residuals=np.load(SAVEPATH+'Fits_'+str(int(width))
                                +'/LightCurve_fits_white.npz')['residuals']*10**-6
        for b in range(0,LC_l.shape[1]):
            LC_l[:,b]+=white_residuals
            
    
    
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])
    
    avgF=np.empty([len(time0)/z,len(bin_ctr)])*np.nan
    RMSE_est=np.empty([len(bin_ctr)])*np.nan
    
    beta_arr=np.empty([LC_l.shape[1],9])

    for b in range(0,LC_l.shape[1]):
        if np.isnan(LC_l[0,b])==True or np.isnan(LC_l[1,b])==True or np.isnan(LC_l[2,b])==True:
            print '---------', bin_ctr[b],'---------'
            continue
        for f in range(0,LC_l.shape[0]):
            if LC_l[f,b]<low or LC_l[f,b]>high:
                LC_l[f,b]=np.nan
     
        LCb=LC_l[:,b]
        
        if noise_red==True:
            n_exp=len(time0)
            model_inputs=np.load(SAVEPATH+'NoiseModel_Inputs_'+str(int(width))+'.npz')
            X_loc=((model_inputs['binned_x'])[:,b]-(model_inputs['binned_x'])[0,b]).reshape(-1,1)
            X_sq=(X_loc*X_loc)
            Y_loc=(model_inputs['yshift']).reshape(-1,1)
            Y_sq=(Y_loc*Y_loc)

            XY=(X_loc*Y_loc)

            bg=((model_inputs['binned_bg'])[:,b]).reshape(-1,1)

            airmass=(np.load(SAVEPATH+'HeaderData.npz')['airmass']).reshape(-1,1)
            median_fwhm=(np.load(SAVEPATH+'FlattenedSpectra.npz')['fwhm_av'])[0,:].reshape(-1,1)
            ones=(np.ones(n_exp)).reshape(-1,1)

            time_fl=(np.copy(time0)).reshape(-1,1)



            model_stack_full=np.hstack((ones,airmass,median_fwhm,X_loc,Y_loc,bg,X_sq,Y_sq,XY))

            out_index=np.array([],dtype=int)
            for t in range(0,len(time0)):
                if time0[t]<timein or time0[t]>timeeg:
                    out_index=np.append(out_index,int(t))
            n_out=len(out_index)  

            #model_stack_oott=np.ones([n_out,model_stack_full.shape[1]])
            model_stack_oott=model_stack_full[out_index,:]
    
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
    
        avgF[:,b]=oot_F0/out0
        
        if noise_red==True:
            X_arr=model_stack_oott
            LC_oott=np.append(new[np.where(time0<timein),b],new[np.where(time0>timeeg),b])
            beta_arr[b,:]=np.dot(np.dot(inv(np.dot(X_arr.T,X_arr)),X_arr.T),LC_oott)

            noise_model_fit=np.dot(model_stack_full,beta_arr[b,:])
        

        if noise_red==True:
            fig,ax=plt.subplots(1,3,figsize=(15,4))
            ax[0].plot(time0,LCb,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.2)
            ax[0].plot(oot_t,oot_F,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[0].plot(time0,out,'-',color='black')
            #ax[0].set_ylim(0.97,np.nanmax(LCb))
            ax[0].set_title(str(int(bin_ctr[b])))
            ax[0].set_xlabel('Time,[hrs]')
            ax[0].set_ylabel('Relative Flux [hrs]')
            
            ax[1].plot(time0,new[:,b],'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.2)
            ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[1].plot(time0,noise_model_fit,'-',color='black')
            #ax[0].set_ylim(0.97,np.nanmax(LCb))
            ax[1].set_title(str(int(bin_ctr[b])))
            ax[1].set_xlabel('Time,[hrs]')
            ax[1].set_ylabel('Relative Flux [hrs]')
            
            new[:,b]=new[:,b]/noise_model_fit
            
            ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
            Fa=(np.append(new[np.where(time0<timein),b],new[np.where(time0>timeeg),b]))
        
            ta=ta[~np.isnan(Fa)]
            Fa=Fa[~np.isnan(Fa)]

            RMSE_est[b]=np.sqrt(np.nanmean(((Fa)-1.0)**2.))

            ax[2].plot(time0,new[:,b],'.',markersize=9.
                       ,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.5)
            #ax[2].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[2].plot(time0,out/out,'-',color='black')
            ax[2].set_title(str(np.round(RMSE_est[b],6)),fontsize=15)
            ax[2].set_ylim(ybot,ytop)
            ax[2].set_xlabel('Time,[hrs]')
            ax[2].set_ylabel('Relative Flux [hrs]')
            plt.show()
            
        else:
            fig,ax=plt.subplots(1,2,figsize=(10,4))
            ax[0].plot(time0,LCb,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.2)
            ax[0].plot(oot_t,oot_F,'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[0].plot(time0,out,'-',color='black')
            #ax[0].set_ylim(0.97,np.nanmax(LCb))
            ax[0].set_title(str(int(bin_ctr[b])))
            ax[0].set_xlabel('Time,[hrs]')
            ax[0].set_ylabel('Relative Flux [hrs]')
           
            
            ta=(np.append(time0[np.where(time0<timein)],time0[np.where(time0>timeeg)]))
            Fa=(np.append(new[np.where(time0<timein),b],new[np.where(time0>timeeg),b]))
        
            ta=ta[~np.isnan(Fa)]
            Fa=Fa[~np.isnan(Fa)]

            RMSE_est[b]=np.sqrt(np.nanmean(((Fa)-1.0)**2.))

            ax[1].plot(time0,new[:,b],'.',markersize=9.,color=scal_m.to_rgba(bin_ctr[b]),alpha=0.2)
            ax[1].plot(oot_t0,oot_F0/out0,'.',markersize=11.,color=scal_m.to_rgba(bin_ctr[b]+width))
            ax[1].plot(time0,out/out,'-',color='black')
            ax[1].set_title(str(np.round(RMSE_est[b],6)),fontsize=15)
            ax[1].set_ylim(ybot,ytop)
            ax[1].set_xlabel('Time,[hrs]')
            ax[1].set_ylabel('Relative Flux [hrs]')
            plt.show()
            
    #ones,airmass,median_fwhm,X_loc,Y_loc,bg,X_sq,Y_sq,XY
    fig,ax=plt.subplots(3,3,figsize=(15,12))
    ax[0,0].plot(bin_ctr,beta_arr[:,3],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[0,0].set_title('X_shift')

    ax[0,1].plot(bin_ctr,beta_arr[:,4],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[0,1].set_title('Y_shift')

    ax[0,2].plot(bin_ctr,beta_arr[:,5],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[0,2].set_title('Background Counts')
    ###
    ax[1,0].plot(bin_ctr,beta_arr[:,6],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[1,0].set_title('X^2')

    ax[1,1].plot(bin_ctr,beta_arr[:,7],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[1,1].set_title('Y^2')

    ax[1,2].plot(bin_ctr,beta_arr[:,8],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[1,2].set_title('XY')
    ###
    ax[2,0].plot(bin_ctr,beta_arr[:,1],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[2,0].set_title('Z')

    ax[2,1].plot(bin_ctr,beta_arr[:,2],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[2,1].set_title('FWHM')

    ax[2,2].plot(bin_ctr,beta_arr[:,0],linewidth=4.0,color='slateblue',alpha=0.7)
    ax[2,2].set_title('1')

    plt.show()

    if corr==True:
        np.savez_compressed(SAVEPATH+'LC_bins_br_'+str(int(width))+'_Corr.npz',
                            data=new,time=time0,rmse=RMSE_est,bin_ctr=bin_ctr,
                            err_t=err_t,err_p=err_p,avt=oot_t0,avf=avgF,beta=beta_arr)
    else:
        np.savez_compressed(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz',
                            data=new,time=time0,rmse=RMSE_est,bin_ctr=bin_ctr,
                            err_t=err_t,err_p=err_p,avt=oot_t0,avf=avgF,beta=beta_arr)