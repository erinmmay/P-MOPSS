# Noise Model Input Prep #
from mF_Binning_Model import BinFunc_t
import numpy as np

np_load_special = lambda *a,**k: np.load(*a, allow_pickle=True, **k)

import matplotlib.pyplot as plt
import matplotlib

import spectres


def NoiseModel_Inputs(SAVEPATH,sw, ew, sb, eb, bin_width, obj_skip, CON, ap0):
    ## Noise Model Prep ### 

    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(bin_width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['bin_centers']    
    print(bin_ctr)

    n_bins=len(bin_ctr)
    width=bin_width#bin_ctr[1]-bin_ctr[0]
    start=sb
    end=eb
    print(start,end,width,n_bins)
    
    bin_ctr=np.append(bin_ctr,bin_ctr[-1]+width)

    ##############

    ShiftSpec=np.load(SAVEPATH+'ShiftedSpec_All_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')
    FlatSpec=np.load(SAVEPATH+'FlattenedSpectra_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')

    wave=ShiftSpec['wave']
    print(wave.shape)

    n_exp=wave.shape[1]
    n_obj=wave.shape[0]
    print(n_exp,n_obj)

    save_binns_x=np.empty([n_obj,n_exp,n_bins+1])*np.nan
    save_binns_bg=np.empty([n_obj,n_exp,n_bins+1])*np.nan

    save_white_x=np.empty([n_obj,n_exp,2])*np.nan
    save_white_bg=np.empty([n_obj,n_exp,2])*np.nan

    ysft=ShiftSpec['yshift']

    ###############

    for o in range(0,n_obj):
        if o in obj_skip:
            continue
        print(o)
        o=int(o)
        
        GAUS_fit = np.load(SAVEPATH+'2DFit_Obj'+str(int(o))+'.npz')
        GAUS_params = np_load_special(SAVEPATH+'2DFit_Obj'+str(int(o))+'.npz')
        
#         gaus=FlatSpec['gaus_params'][o,:,:,1]  #1=x-shift
#         bkgd=FlatSpec['gaus_params'][o,:,:,4]  #1=bg counts

        ed_t = GAUS_params['params'][3]
        print(ed_t)
    
        gaus = GAUS_fit['x_fit']
        gaus -= 1.0  #convert pixel value to index value (start at 1 -> start at 0)
        
        bkgd_load = GAUS_fit['background']
        xwidth = bkgd_load.shape[2]
        bkgd = np.nansum(bkgd_load[:,:,ed_t:xwidth-ed_t], axis = 2)
        print(bkgd.shape)

        waves0=np.flip(wave[o,:,:],axis=1)
        xs0=np.flip(gaus,axis=1)
        bg0=np.flip(bkgd,axis=1)

#         waves0 = wave[o,:,:]
#         xs0 = gaus
#         bg0 = bkgd

        #binned_x=np.empty([n_exp,n_bins])*np.nan
        #binned_bg=np.empty([n_exp,n_bins])*np.nan

        #################
        
        ed=0
        lowi=np.argmin(np.abs(waves0[0,:]-(start-width/2)))
        uppi=np.argmin(np.abs(waves0[0,:]-(end+3*width/2)))
        lowi=np.nanmax([lowi-ed,0])
        uppi=np.nanmin([uppi+ed,xs0.shape[1]])
        print(lowi, uppi)
        print(waves0[0,lowi-ed], waves0[0,uppi+ed])

        white_bin = sw+(ew-sw)/2
        bin_ctr_white = np.array([white_bin,white_bin+(ew-sw)])
        
        ed=0
        Wlowi=np.argmin(np.abs(waves0[0,:]-(sw-(ew-sw)/2)))
        Wuppi=np.argmin(np.abs(waves0[0,:]-(ew+3*(ew-sw)/2)))
        Wlowi=np.nanmax([Wlowi-ed,0])
        Wuppi=np.nanmin([Wuppi+ed,xs0.shape[1]])
        print(Wlowi, Wuppi)
        print(waves0[0,Wlowi-ed], waves0[0,Wuppi+ed])
        
        for t in range(0,n_exp):
            save_white_x[o,t,:]  = spectres.spectres(bin_ctr_white, waves0[t,Wlowi-ed:Wuppi+ed], xs0[t,Wlowi-ed:Wuppi+ed])
            save_white_bg[o,t,:] = spectres.spectres(bin_ctr_white, waves0[t,Wlowi-ed:Wuppi+ed], bg0[t,Wlowi-ed:Wuppi+ed])
            
            save_binns_x[o,t,:]  = spectres.spectres(bin_ctr, waves0[t,lowi-ed:uppi+ed], xs0[t,lowi-ed:uppi+ed])
            save_binns_bg[o,t,:] = spectres.spectres(bin_ctr, waves0[t,lowi-ed:uppi+ed], bg0[t,lowi-ed:uppi+ed])
            
#         __,save_white_x[o,:,:]=(BinFunc_t(xs0,waves0,start,end,end-start,True))
#         __,save_white_bg[o,:,:]=(BinFunc_t(bg0,waves0,start,end,end-start,True))

#         __,save_binns_x[o,:,:]=BinFunc_t(xs0,waves0,start,end,width,True)
#         __,save_binns_bg[o,:,:]=BinFunc_t(bg0,waves0,start,end,width,True)
        
        ####
        plt.figure((10*o)+1,figsize=(15,4))
        plt.subplots_adjust(bottom=0.12,left=0.12)
        plt.clf()

        plt.plot(waves0[0,:],bg0[0,:]/np.nanmax(bg0[0,lowi-ed:uppi+ed]),color='black')
        plt.plot(bin_ctr,save_binns_bg[o,0,:]/np.nanmax(save_binns_bg[o,0,:]),'.',markersize=10,color='red')
        for b in bin_ctr:
            plt.axvline(x=b-width/2.,color='grey',linewidth=0.5)
        plt.axvline(x=bin_ctr[-1]+width/2.,color='grey',linewidth=0.5)
        #ax[0].set_ylim(-0.1,1.4)
        plt.xlim(3000,10000)
                   
        plt.xlabel('Wavelength, [$\AA$]',fontsize=15,ha='center',va='top')
        plt.ylabel('Relative BG',fontsize=15,ha='right',va='center',rotation='vertical')
        plt.figtext(0.15,0.80,'obj'+str(int(o))+'_bg',fontsize=25,color='red')
        plt.show()
        plt.close((10*o)+1)
        
        #####
        plt.figure((10*o)+2,figsize=(15,4))
        plt.subplots_adjust(bottom=0.12,left=0.12)
        plt.clf()

        plt.plot(waves0[0,:],xs0[0,:]/np.nanmax(xs0[0,lowi-ed:uppi+ed]),color='black')
        plt.plot(bin_ctr,save_binns_x[o,0,:]/np.nanmax(save_binns_x[o,0,:]),'.',markersize=10,color='red')
        for b in bin_ctr:
            plt.axvline(x=b-width/2.,color='grey',linewidth=0.5)
        plt.axvline(x=bin_ctr[-1]+width/2.,color='grey',linewidth=0.5)
        #ax[0].set_ylim(-0.1,1.4)
        plt.xlim(3000,10000)
                   
        plt.xlabel('Wavelength, [$\AA$]',fontsize=15,ha='center',va='top')
        plt.ylabel('Relative X',fontsize=15,ha='right',va='center',rotation='vertical')
        plt.figtext(0.15,0.80,'obj'+str(int(o))+'_X_shift',fontsize=25,color='red')
        plt.show()
        plt.close((10*o)+2)
        
        ####
            

    np.savez_compressed(SAVEPATH+
                        'NoiseModel_Inputs_'+str(int(bin_width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz',
                        white_x=save_white_x[:,:,0],white_bg=save_white_bg[:,:,0],
                       binned_x=save_binns_x,binned_bg=save_binns_bg, yshift=ysft)