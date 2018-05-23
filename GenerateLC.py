import numpy as np

import matplotlib
import matplotlib.pyplot as plt


from setup import *

def LCgen_white(SAVEPATH,corr,Cals_ind,csn):
    es=1.
    
    if corr==True:
        Data=np.load(SAVEPATH+'Binned_Data_White_Corr.npz')['bin_counts']
        errsw_t=np.load(SAVEPATH+'Binned_Data_White_Corr.npz')['bin_err']
        errsw_p=np.load(SAVEPATH+'Binned_Data_White_Corr.npz')['bin_ptn']
    else:
        Data=np.load(SAVEPATH+'Binned_Data_White.npz')['bin_counts']
        errsw_t=np.load(SAVEPATH+'Binned_Data_White.npz')['bin_err']
        errsw_p=np.load(SAVEPATH+'Binned_Data_White.npz')['bin_ptn']

    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    n_exp=len(time0)
         
        

#W52- 2,3,5,7,8
    #Cals_ind=np.array([2,3,5,8])
    #csn=2
    
    Cals_ind=Cals_ind
    csn=csn
    
    if len(Cals_ind)==0:
        Cals=np.ones_like(Data[:,0,0])
        errs_cw_t=np.zeros_like(Data[:,0,0])
        errs_cw_p=np.zeros_like(Data[:,0,0])
    else:
        Cals=np.zeros_like(Data[:,0,0])
        errs_cw_t=np.zeros_like(Data[:,0,0])
        errs_cw_p=np.zeros_like(Data[:,0,0])
        for c in Cals_ind:
            Cals=(np.nansum([Cals,Data[:,0,c]],axis=0))
            errs_cw_t=np.sqrt(np.nansum([errs_cw_t,errsw_t[:,0,c]**2.],axis=0))
            errs_cw_p=np.sqrt(np.nansum([errs_cw_p,errsw_p[:,0,c]**2.],axis=0))


    errs_cw_t/=np.nanmean(Cals)
    errs_cw_p/=np.nanmean(Cals)


    Cals=Cals/np.nanmean(Cals)

    LC=(Data[:,0,0]/Cals)
    CS=(Data[:,0,csn]/Cals)

    errs_w_t=np.sqrt(np.nansum([(errsw_t[:,0,0]/Data[:,0,0])**2.,(errs_cw_t/Cals)**2.],axis=0))*LC
    errs_w_p=np.sqrt(np.nansum([(errsw_p[:,0,0]/Data[:,0,0])**2.,(errs_cw_p/Cals)**2.],axis=0))*LC

    errs_cs_t=np.sqrt(np.nansum([(errsw_t[:,0,0]/Data[:,0,csn])**2.,(errs_cw_t/Cals)**2.],axis=0))*CS
    errs_cs_p=np.sqrt(np.nansum([(errsw_p[:,0,0]/Data[:,0,csn])**2.,(errs_cw_p/Cals)**2.],axis=0))*CS

    errs_w_t/=np.nanmean(LC[0:20])
    errs_w_p/=np.nanmean(LC[0:20])

    errs_cs_t/=np.nanmean(CS[0:20])
    errs_cs_p/=np.nanmean(CS[0:20])

    LC=LC/np.nanmean(LC[0:20])
    CS=CS/np.nanmean(CS[0:20])
    LCd=LC/CS


    errs_lcd_w_t=np.sqrt(np.nansum([(errs_w_t/LC)**2.,(errs_cs_t/CS)**2.],axis=0))*LCd
    errs_lcd_w_p=np.sqrt(np.nansum([(errs_w_p/LC)**2.,(errs_cs_p/CS)**2.],axis=0))*LCd

    for t in range(0,len(LC)):
        if LC[t]<0.8 or LC[t]>1.2:
            LC[t]=np.nan
            CS[t]=np.nan
            LCd[t]=np.nan
    for t in range(0,len(LC)):
        if t>1 and t<len(LC)-1:
            if np.isfinite(LC[t])==False:
                LC[t]=np.nanmedian(np.append(LC[t-1],LC[t+1]))
            if np.isfinite(CS[t])==False:
                CS[t]=np.nanmedian(np.append(CS[t-1],CS[t+1]))
            if np.isfinite(LCd[t])==False:
                LCd[t]=np.nanmedian(np.append(LCd[t-1],LCd[t+1]))
                
    LC=LC/np.nanmean(LC[0:20])
    CS=CS/np.nanmean(CS[0:20])     
    LCd=LC/CS

    errs_lcd_w_t/=np.nanmean(LCd[0:20])
    errs_lcd_w_p/=np.nanmean(LCd[0:20])

    ymax_lc=np.nanmax(LC)
    ymin_lc=np.nanmin(LC)
    
    ymax_cs=np.nanmax(CS)
    ymin_cs=np.nanmin(CS)


    fig,ax=plt.subplots(1,3,figsize=(15,4))

    ax[0].plot(time0,LC,'.',color='grey')
    ax[0].errorbar(time0,LC,yerr=es*errs_w_t,ecolor='grey',elinewidth=0.5,alpha=0.5,zorder=9,fmt=None)
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
    ax[2].errorbar(time0,LCd,yerr=es*errs_lcd_w_t,ecolor='grey',elinewidth=0.5,alpha=0.5,zorder=9,fmt=None)
    #ax[2].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
    ax[2].set_title('Divided - WHITE')
    ax[2].set_xlabel('Time,[days]')
    #ax[1].set_ylabel('Relative Flux')

    plt.show()
    if corr==True:    
        np.savez_compressed(SAVEPATH+'LCwhite_Corr.npz',data=LC,cs=CS,time=time0,err_t=errs_w_t,err_p=errs_w_p)
    else:
        np.savez_compressed(SAVEPATH+'LCwhite.npz',data=LC,cs=CS,time=time0,err_t=errs_w_t,err_p=errs_w_p)
    return
        
def LCgen_binns(SAVEPATH,width,corr,Cals_ind,csn):
    es=1.
    
    if corr==True:
        Datal=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'_Corr.npz')['bin_counts']
        errs_t=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'_Corr.npz')['bin_err']
        errs_p=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'_Corr.npz')['bin_ptn']
    else:
        Datal=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_counts']
        errs_t=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_err']
        errs_p=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_ptn']
    
    time0=np.load(SAVEPATH+'Obs_times.npz')['times']
    
    bin_arr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bins']
    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(width))+'.npz')['bin_centers']
    width=bin_arr[1]-bin_arr[0]

    print bin_arr
    
    bin_arr=np.append(bin_arr,[bin_arr[-1]+width,bin_arr[-1]+2*width])

    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])

    Cals_l=np.empty([Datal.shape[0],Datal.shape[1]])*0.0
    LC_l=np.zeros_like(Cals_l)
    CS_l=np.zeros_like(Cals_l)
    errs_cl_t=np.zeros_like(Cals_l)
    errs_cl_p=np.zeros_like(Cals_l)
    errs_l_t=np.zeros_like(Cals_l)
    errs_l_p=np.zeros_like(Cals_l)
    errs_cs_l_t=np.zeros_like(Cals_l)
    errs_cs_l_p=np.zeros_like(Cals_l)
    errs_lcd_l_t=np.zeros_like(Cals_l)
    errs_lcd_l_p=np.zeros_like(Cals_l)
    LC_d=np.zeros_like(Cals_l)


    Cals_l=np.zeros_like(Datal[:,:,0])
    errs_cl_t=np.zeros_like(Datal[:,:,0])
    errs_cl_p=np.zeros_like(Datal[:,:,0])
    for b in range(0,Datal.shape[1]):
        if len(Cals_ind)==0:
            Cals_l[:,b]=np.ones_like(Datal[:,0,0])
        else:
            for c in Cals_ind:
                Cals_l[:,b]=(np.nansum([Cals_l[:,b],Datal[:,b,c]],axis=0))
                errs_cl_t[:,b]=np.sqrt(np.nansum([errs_cl_t[:,b],errs_t[:,b,c]**2.],axis=0))
                errs_cl_p[:,b]=np.sqrt(np.nansum([errs_cl_p[:,b],errs_p[:,b,c]**2.],axis=0))

        errs_cl_t/=np.nanmean(Cals_l[:,b])
        errs_cl_p/=np.nanmean(Cals_l[:,b])
    
        Cals_l[:,b]=Cals_l[:,b]/np.nanmean(Cals_l[:,b])
    
    ###
    
        LC_l[:,b]=(Datal[:,b,0]/Cals_l[:,b])
        CS_l[:,b]=(Datal[:,b,csn]/Cals_l[:,b])
        
        errs_l_t[:,b]=np.sqrt(np.nansum([(errs_t[:,b,0]/Datal[:,b,0])**2.,(errs_cl_t[:,b]/Cals_l[:,b])**2.],axis=0))*LC_l[:,b]
        errs_l_p[:,b]=np.sqrt(np.nansum([(errs_p[:,b,0]/Datal[:,b,0])**2.,(errs_cl_p[:,b]/Cals_l[:,b])**2.],axis=0))*LC_l[:,b]
    
        errs_cs_l_t[:,b]=np.sqrt(np.nansum([(errs_t[:,b,0]/Datal[:,b,csn])**2.,(errs_cl_t[:,b]/Cals_l[:,b])**2.],axis=0))*LC_l[:,b]
        errs_cs_l_p[:,b]=np.sqrt(np.nansum([(errs_p[:,b,0]/Datal[:,b,csn])**2.,(errs_cl_p[:,b]/Cals_l[:,b])**2.],axis=0))*LC_l[:,b]
    
        errs_l_t[:,b]/=np.nanmean(LC_l[0:20,b])
        errs_l_p[:,b]/=np.nanmean(LC_l[0:20,b])
    
        errs_cs_l_t[:,b]/=np.nanmean(CS_l[0:20,b])
        errs_cs_l_p[:,b]/=np.nanmean(CS_l[0:20,b])
    
        LC_l[:,b]=LC_l[:,b]/np.nanmean(LC_l[0:20,b])
        CS_l[:,b]=CS_l[:,b]/np.nanmean(CS_l[0:20,b])
        LC_d[:,b]=LC_l[:,b]/CS_l[:,b]

        errs_lcd_l_t[:,b]=np.sqrt(np.nansum([(errs_l_t[:,b]/LC_l[:,b])**2.,(errs_cs_l_t[:,b]/CS_l[:,b])**2.],axis=0))*LC_d[:,b]
        errs_lcd_l_p[:,b]=np.sqrt(np.nansum([(errs_l_p[:,b]/LC_l[:,b])**2.,(errs_cs_l_p[:,b]/CS_l[:,b])**2.],axis=0))*LC_d[:,b]
    
        for t in range(0,LC_l.shape[0]):
            if LC_l[t,b]<0.8:
                LC_l[t,b]=np.nan
                CS_l[t,b]=np.nan
                LC_d[t,b]=np.nan
            
        LC_l[:,b]=LC_l[:,b]/np.nanmean(LC_l[0:20,b])
        CS_l[:,b]=CS_l[:,b]/np.nanmean(CS_l[0:20,b])
        LC_d[:,b]=LC_l[:,b]/CS_l[:,b]
    
        errs_lcd_l_t[:,b]/=np.nanmean(LC_d[0:20,b])
        errs_lcd_l_p[:,b]/=np.nanmean(LC_d[0:20,b])
        
    
        ymax_lc=np.nanmax(LC_l[:,b])
        ymin_lc=np.nanmin(LC_l[:,b])
    
        ymax_cs=np.nanmax(CS_l[:,b])
        ymin_cs=np.nanmin(CS_l[:,b])

        fig,ax=plt.subplots(1,3,figsize=(15,4))
        ax[0].plot(time0*24.,LC_l[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        ax[0].errorbar(time0*24.,LC_l[:,b],yerr=es*errs_l_t[:,b],ecolor=scal_m.to_rgba(bin_ctr[b]),elinewidth=0.5,alpha=0.5,zorder=9,fmt=None)
        #ax[0].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
        ax[0].set_title(str(int(bin_ctr[b])))
        ax[0].set_xlabel('Time,[hrs]')
        ax[0].set_ylabel('Relative Flux')
    
        ax[1].plot(time0*24.,CS_l[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        #ax[1].set_ylim(ymin_cs-0.01,ymax_cs+0.01)
        ax[1].set_title('Check Star - ' +str(int(bin_ctr[b])))
        ax[1].set_xlabel('Time,[hrs]')
    #ax[1].set_ylabel('Relative Flux')
    
        ax[2].plot(time0*24.,LC_d[:,b],'.',color=scal_m.to_rgba(bin_ctr[b]))
        ax[2].errorbar(time0*24.,LC_d[:,b],yerr=es*errs_lcd_l_t[:,b],ecolor=scal_m.to_rgba(bin_ctr[b]),elinewidth=0.5,alpha=0.5,zorder=9,fmt=None)
        #ax[2].set_ylim(ymin_lc-0.01,ymax_lc+0.01)
        ax[2].set_title('Divided - ' +str(int(bin_ctr[b])))
        ax[2].set_xlabel('Time,[hrs]')
        #ax[1].set_ylabel('Relative Flux')
    
        plt.show()

    if corr==True:    
        np.savez_compressed(SAVEPATH+'LC_bins_'+str(int(width))+'_Corr.npz',data=LC_l,cs=CS_l,time=time0,bin_ctr=bin_ctr,err_t=errs_l_t,err_p=errs_l_p)
    else:    
        np.savez_compressed(SAVEPATH+'LC_bins_'+str(int(width))+'.npz',data=LC_l,cs=CS_l,time=time0,bin_ctr=bin_ctr,err_t=errs_l_t,err_p=errs_l_p)
    return
      