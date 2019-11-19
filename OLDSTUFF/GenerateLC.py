import numpy as np

import matplotlib
import matplotlib.pyplot as plt


from setup import *

def LCgen_white(SAVEPATH,corr,Cals_ind,csn):
    es=1.
    
    if corr==True:
        Data=np.load(SAVEPATH+'NoiseModel_FitResults_White.npz')['data']
        Data_c=(np.load(SAVEPATH+'Binned_Data_White.npz')['bin_counts'])[:,:,0]
    else:
        Data=(np.load(SAVEPATH+'AirmassRemove_White.npz')['data'])
        Data_c=(np.load(SAVEPATH+'Binned_Data_White.npz')['bin_counts'])[:,:,0]
        airmass_func=Data/Data_c
    errsw_t=(np.load(SAVEPATH+'Binned_Data_White.npz')['bin_err'])[:,:,0]
    errsw_p=(np.load(SAVEPATH+'Binned_Data_White.npz')['bin_ptn'])[:,:,0]
    
   

    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    n_exp=len(time0)
         
        
    print(Data.shape)
    #object, time, bin
    #old: time, bin, object
#W52- 2,3,5,7,8
    #Cals_ind=np.array([2,3,5,8])
    #csn=2
    
    Cals_ind=Cals_ind
    csn=csn
    
    if len(Cals_ind)==0:
        Cals=np.ones_like(Data[0,:])
        errs_cw_t=np.ones_like(Data[0,:])*0.0
        errs_cw_p=np.ones_like(Data[0,:])*0.0
    else:
        Cals=np.zeros_like(Data[0,:])*np.nan
        errs_cw_t=np.ones_like(Data[0,:])*np.nan
        errs_cw_p=np.ones_like(Data[0,:])*np.nan
        for c in Cals_ind:
            Cals=(np.nansum([Cals,Data[c,:]],axis=0))
            errs_cw_t=(np.nansum([errs_cw_t,Data_c[c,:]],axis=0))
    
    errs_cw_t=1./errs_cw_t
    #print errs_cw_t
    errs_cw_t+=(1/Data_c[0,:])
    #print errs_cw_t
    errs_cw_p=np.sqrt(errs_cw_t)
    #print errs_cw_p
    LC=(Data[0,:]/Cals)
    CS=(Data[csn,:]/Cals)

    errs_cw_p*=LC

    LCd=LC/CS
    
    print(np.nanmedian(errs_cw_p)*10**6.)


#    errs_lcd_w_t=np.sqrt(np.nansum([(errs_w_t/LC)**2.,(errs_cs_t/CS)**2.],axis=0))*LCd
#    errs_lcd_w_p=np.sqrt(np.nansum([(errs_w_p/LC)**2.,(errs_cs_p/CS)**2.],axis=0))*LCd

#     for t in range(0,len(LC)):
#         if LC[t]<0.8 or LC[t]>1.2:
#             LC[t]=np.nan
#             CS[t]=np.nan
#             LCd[t]=np.nan
#     for t in range(0,len(LC)):
#         if t>1 and t<len(LC)-1:
#             if np.isfinite(LC[t])==False:
#                 LC[t]=np.nanmedian(np.append(LC[t-1],LC[t+1]))
#             if np.isfinite(CS[t])==False:
#                 CS[t]=np.nanmedian(np.append(CS[t-1],CS[t+1]))
#             if np.isfinite(LCd[t])==False:
#                 LCd[t]=np.nanmedian(np.append(LCd[t-1],LCd[t+1]))
                
    ymax_lc=np.nanmax(LC)
    ymin_lc=np.nanmin(LC)
    
    ymax_cs=np.nanmax(CS)
    ymin_cs=np.nanmin(CS)


    fig,ax=plt.subplots(1,3,figsize=(15,4))

    ax[0].plot(time0,LC,'.',color='grey')
    ax[0].errorbar(time0,LC,yerr=es*errs_cw_t,ecolor='grey',elinewidth=0.5,alpha=0.5,zorder=9,fmt='none')
    #ax[0].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
    ax[0].set_title('WHITE')
    ax[0].set_xlabel('Time,[days]')
    ax[0].set_ylabel('Relative Flux')

    ax[1].plot(time0,CS,'.',color='grey')
    #ax[1].set_ylim(ymin_cs-0.01,ymax_cs+0.01)
    ax[1].set_title('Check Star - WHITE')
    ax[1].set_xlabel('Time,[days]')
    #ax[1].set_ylabel('Relative Flux')

    ax[2].plot(time0,LCd,'.',color='grey')
    #ax[2].errorbar(time0,LCd,yerr=es*errs_lcd_w_t,ecolor='grey',elinewidth=0.5,alpha=0.5,zorder=9,fmt=None)
    #ax[2].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
    ax[2].set_title('Divided - WHITE')
    ax[2].set_xlabel('Time,[days]')
    #ax[1].set_ylabel('Relative Flux')

    plt.show()
    np.savez_compressed(SAVEPATH+'LCwhite.npz',data=LC,cs=CS,time=time0,err_t=errs_cw_t,err_p=errs_cw_p)
    return
        
def LCgen_binns(SAVEPATH,width,corr,Cals_ind,csn):
    es=1.
    
    if corr==True:
        Datal=np.load(SAVEPATH+'NoiseModel_FitResults_'+str(int(width))+'.npz')['data']
    else:
        Datal=np.load(SAVEPATH+'AirmassRemove_'+str(int(width))+'.npz')['data']
        Datal_c=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_counts']
    errs_t=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_err']
    errs_p=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_ptn']
    
    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    
    bin_arr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bins']
    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_centers']
    width=bin_arr[1]-bin_arr[0]

    print(bin_arr)
    
    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-1]+2*width])

    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])

    Cals_l=np.empty([Datal.shape[1],Datal.shape[2]])*np.nan
    LC_l=np.zeros_like(Cals_l)*np.nan
    CS_l=np.zeros_like(Cals_l)*np.nan
    errs_cl_t=np.zeros_like(Cals_l)*np.nan
    errs_cl_p=np.zeros_like(Cals_l)*np.nan
    errs_l_t=np.zeros_like(Cals_l)*np.nan
    errs_l_p=np.zeros_like(Cals_l)*np.nan
    errs_cs_l_t=np.zeros_like(Cals_l)*np.nan
    errs_cs_l_p=np.zeros_like(Cals_l)*np.nan
    errs_lcd_l_t=np.zeros_like(Cals_l)*np.nan
    errs_lcd_l_p=np.zeros_like(Cals_l)*np.nan
    LC_d=np.zeros_like(Cals_l)*np.nan

    print(Datal.shape)
    
    Cals_l=np.ones_like(Datal[0,:,:])*np.nan
    errs_cl_t=np.ones_like(Datal[0,:,:])*np.nan
    errs_cl_p=np.ones_like(Datal[0,:,:])*np.nan
    for b in range(0,Datal.shape[2]):
        if len(Cals_ind)==0:
            Cals_l[:,b]=np.ones_like(Datal[0,:,0])*1.0
            errs_cl_t[:,b]=np.ones_like(Datal[0,:,0])*0.0
            errs_cl_p[:,b]=np.ones_like(Datal[0,:,0])*0.0
        else:
            for c in Cals_ind:
                Cals_l[:,b]=(np.nansum([Cals_l[:,b],Datal[c,:,b]],axis=0))
                errs_cl_t[:,b]=(np.nansum([errs_cl_t[:,b],Datal_c[c,:,b]],axis=0))
              
        errs_cl_t[:,b]=1./errs_cl_t[:,b]
        errs_cl_t[:,b]+=(1/Datal_c[0,:,b])
        errs_cl_p[:,b]=np.sqrt(errs_cl_t[:,b])
        #print errs_cl_p[:,b]
        
        LC_l[:,b]=(Datal[0,:,b]/Cals_l[:,b])
        CS_l[:,b]=(Datal[csn,:,b]/Cals_l[:,b])
                               
        errs_cl_p[:,b]*=LC_l[:,b]
       
        LC_d[:,b]=LC_l[:,b]/CS_l[:,b]
        
        print(np.nanmedian(errs_cl_p[:,b])*10**6.)
        
    
        ymax_lc=np.nanmax(LC_l[:,b])
        ymin_lc=np.nanmin(LC_l[:,b])
    
        ymax_cs=np.nanmax(CS_l[:,b])
        ymin_cs=np.nanmin(CS_l[:,b])

        fig,ax=plt.subplots(1,3,figsize=(15,4))
        ax[0].plot(time0*24.,LC_l[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        ax[0].errorbar(time0*24.,LC_l[:,b],yerr=es*errs_cl_t[:,b],
                       ecolor=scal_m.to_rgba(bin_ctr[b]),elinewidth=0.5,alpha=0.5,zorder=9,fmt='none')
        #ax[0].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
        ax[0].set_title(str(int(bin_ctr[b])))
        ax[0].set_xlabel('Time,[hrs]')
        ax[0].set_ylabel('Relative Flux')
    
        ax[1].plot(time0*24.,CS_l[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        ax[1].set_ylim(ymin_cs-0.01,ymax_cs+0.01)
        ax[1].set_title('Check Star - ' +str(int(bin_ctr[b])))
        ax[1].set_xlabel('Time,[hrs]')
    #ax[1].set_ylabel('Relative Flux')
    
        ax[2].plot(time0*24.,LC_d[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        #ax[2].errorbar(time0*24.,LC_d[:,b],yerr=es*errs_lcd_l_t[:,b],
        #               ecolor=scal_m.to_rgba(bin_ctr[b]),elinewidth=0.5,alpha=0.5,zorder=9,fmt=None)
        #ax[2].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
        ax[2].set_title('Divided - ' +str(int(bin_ctr[b])))
        ax[2].set_xlabel('Time,[hrs]')
        #ax[1].set_ylabel('Relative Flux')
    
        plt.show()

    np.savez_compressed(SAVEPATH+'LC_bins_'+str(int(width))+'.npz',
                            data=LC_l,cs=CS_l,time=time0,bin_ctr=bin_ctr,err_t=errs_cl_t,err_p=errs_cl_p)
    return
      