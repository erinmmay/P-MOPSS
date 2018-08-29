# Noise Model Input Prep #
from Binning_Model import BinFunc_t
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


def NoiseModel_Inputs(SAVEPATH,bin_width):
    ## Noise Model Prep ### 

    bin_ctr=np.load(SAVEPATH+'Binned_Data_'+str(int(bin_width))+'.npz')['bin_centers']    
    print bin_ctr

    n_bins=len(bin_ctr)
    width=bin_ctr[1]-bin_ctr[0]
    start=bin_ctr[0]-width/2
    end=bin_ctr[-1]+width/2
    print start,end,width,n_bins

    ##############

    ShiftSpec=np.load(SAVEPATH+'ShiftedSpec_All.npz')
    FlatSpec=np.load(SAVEPATH+'FlattenedSpectra.npz')

    wave=ShiftSpec['wave']
    print wave.shape

    n_exp=wave.shape[1]
    n_obj=wave.shape[0]
    print n_exp,n_obj

    save_binns_x=np.empty([n_obj,n_exp,n_bins])*np.nan
    save_binns_bg=np.empty([n_obj,n_exp,n_bins])*np.nan

    save_white_x=np.empty([n_obj,n_exp,1])*np.nan
    save_white_bg=np.empty([n_obj,n_exp,1])*np.nan

    ysft=ShiftSpec['yshift']

    ###############

    for o in range(0,n_obj):
        print o
        o=int(o)
        gaus=FlatSpec['gaus_params'][o,:,:,1]  #1=x-shift
        bkgd=FlatSpec['gaus_params'][o,:,:,4]  #1=bg counts

        waves0=np.fliplr(wave[o,:,:])
        xs0=np.fliplr(gaus)
        bg0=np.fliplr(bkgd)

        #binned_x=np.empty([n_exp,n_bins])*np.nan
        #binned_bg=np.empty([n_exp,n_bins])*np.nan

        #################
        __,save_white_x[o,:]=(BinFunc_t(xs0,waves0,start,end,end-start,True))
        __,save_white_bg[o,:]=(BinFunc_t(bg0,waves0,start,end,end-start,True))

        __,save_binns_x[o,:]=BinFunc_t(xs0,waves0,start,end,width,True)
        __,save_binns_bg[o,:]=BinFunc_t(bg0,waves0,start,end,width,True)

    np.savez_compressed(SAVEPATH+'NoiseModel_Inputs_'+str(int(bin_width))+'.npz',
                        white_x=save_white_x[:,:,0],white_bg=save_white_bg[:,:,0],
                       binned_x=save_binns_x,binned_bg=save_binns_bg, yshift=ysft)