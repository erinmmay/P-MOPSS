import numpy as np
np.seterr('ignore')

import astropy
import spectres
# import scipy
# from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsec

from setup import *


def BinWhite(SAVEPATH,midtime,start,end,skip,binny):
    load=np.load(SAVEPATH+'ShiftedSpec_All.npz')
    cnt_arr=load['data']
    #cnt_arr=load['convolved']
    wav_arr=load['wave']
    del load
    
    cnt_arr=np.flip(cnt_arr,axis=2)
    wav_arr=np.flip(wav_arr,axis=2)
    
    print(cnt_arr.shape)
    
    n_obj=cnt_arr.shape[0]
    n_exp=cnt_arr.shape[1]
    
    ed=10
    
    #bkgd=np.load(SAVEPATH+'FlattenedSpectra.npz')['flat_bkgd']
    
    ptn_err=np.sqrt(cnt_arr)#+bkgd)

    # defining new wavelength array #
    width_bin=end-start
    numbins=1
    bin_arr=np.linspace(start,end,numbins+1)
    bin_ctr=np.array([])
    for b in range(0,len(bin_arr)-1):
        bin_ctr=np.append(bin_ctr,bin_arr[b]+width_bin/2.)
        
    #bin_ctr=np.append(bin_ctr[0]-width_bin,bin_ctr)
    bin_ctr=np.append(bin_ctr,bin_ctr[-1]+width_bin)
        
    print('  -->> From Lambda=', start, ' to Lambda=', end)
    print('  -->> TOTAL OF ', numbins, 'WAVELENGTH BINS')
    print('       Bin Centers: ', bin_ctr)
    print('       Bin Array:   ', bin_arr)
    print('       Bin Width:   ', width_bin)
        
    bin_cnt=np.empty([n_obj,n_exp,numbins+1])*np.nan
    bin_err=np.empty([n_obj,n_exp,numbins+1])*np.nan

    for o in range(0,n_obj):
        if o in skip:
            continue
        print(' **********', o, '********** ')
        lowi=np.argmin(np.abs(wav_arr[o,0,:]-(start-width_bin/2)))
        uppi=np.argmin(np.abs(wav_arr[o,0,:]-(end+3*width_bin/2)))
        lowi=np.nanmax([lowi-ed,0])
        uppi=np.nanmin([uppi+ed,cnt_arr.shape[2]])
        ed=0
        print(bin_ctr[0]-width_bin/2,bin_ctr[-1]+width_bin/2)
        print((wav_arr[o,0,lowi-ed:uppi+ed])[0],(wav_arr[o,0,lowi-ed:uppi+ed])[-1])
        for t in range(0,n_exp):
            bin_cnt[o,t,:],bin_err[o,t,:]=spectres.spectres(bin_ctr, wav_arr[o,t,lowi-ed:uppi+ed], cnt_arr[o,t,lowi-ed:uppi+ed], spec_errs=ptn_err[o,t,lowi-ed:uppi+ed])
    
        plt.figure(1,figsize=(15,4))
        plt.clf()
        plt.plot(wav_arr[o,10,:],cnt_arr[o,10,:]/np.nanmax(cnt_arr[o,10,:]),color='black')
        plt.plot(bin_ctr,bin_cnt[o,10,:]/np.nanmax(bin_cnt[o,10,:]),'.',markersize=10,color='red')
        plt.errorbar(bin_ctr,bin_cnt[o,10,:]/np.nanmax(bin_cnt[o,10,:]),
                     yerr=10*bin_err[o,10,:]/np.nanmax(bin_cnt[o,10,:]),fmt='None',ecolor='red')
        for b in bin_arr:
            plt.axvline(x=b,color='grey',linewidth=0.5)
        plt.ylim(-0.1,1.4)
        plt.xlim(3000,10000)
        plt.xlabel('Wavelength, [$\AA$]',fontsize=15)
        plt.ylabel('Relative Flux',fontsize=15)
        plt.figtext(0.15,0.80,'obj'+str(int(o)),fontsize=25,color='red')
        plt.show()
        plt.close()
        
    np.savez_compressed(SAVEPATH+'Binned_Data_White.npz',bins=bin_arr,bin_centers=bin_ctr,
                            bin_counts=bin_cnt,bin_err=bin_err,bin_ptn=bin_err)

def BinLam(SAVEPATH,midtime,start,end,width,skip,binny,ed):
    load=np.load(SAVEPATH+'ShiftedSpec_All.npz')
    cnt_arr=load['data']
    wav_arr=load['wave']
    del load
    
    cnt_arr=np.flip(cnt_arr,axis=2)
    wav_arr=np.flip(wav_arr,axis=2)
    
    n_obj=cnt_arr.shape[0]
    n_exp=cnt_arr.shape[1]
    
    #bkgd=np.load(SAVEPATH+'FlattenedSpectra.npz')['flat_bkgd']
    
    ptn_err=np.sqrt(cnt_arr)#+bkgd)

    # defining new wavelength array #
    width_bin=width
    numbins=int((end-start)/width_bin)
    bin_arr=np.linspace(start,end,numbins+1)
    bin_ctr=np.array([])
    for b in range(0,len(bin_arr)-1):
        bin_ctr=np.append(bin_ctr,bin_arr[b]+width_bin/2.)
        
    print('  -->> From Lambda=', start, ' to Lambda=', end)
    print('  -->> TOTAL OF ', numbins, 'WAVELENGTH BINS')
    print('       Bin Centers: ', bin_ctr)
    print('       Bin Array:   ', bin_arr)
    print('       Bin Width:   ', width_bin)
    
    #bin_ctr=np.append(bin_ctr[0]-width_bin,bin_ctr)
    #bin_ctr=np.append(bin_ctr,bin_ctr[-1]+width_bin)
    print(bin_ctr)
        
    bin_cnt=np.empty([n_obj,n_exp,numbins])*np.nan
    bin_err=np.empty([n_obj,n_exp,numbins])*np.nan
    
    ed=ed
    for o in range(0,n_obj):
        if o in skip:
            continue
        print(' **********', o, '********** ')
        lowi=np.argmin(np.abs(wav_arr[o,0,:]-(start-width_bin/2)))
        uppi=np.argmin(np.abs(wav_arr[o,0,:]-(end+3*width_bin/2)))
        lowi=np.nanmax([lowi-ed,0])
        #print uppi, ed, cnt_arr.shape[2]
        uppi=np.nanmin([uppi+ed,cnt_arr.shape[2]])
        
        #ed=0
        print(bin_ctr[0]-width_bin/2,bin_ctr[-1]+width_bin/2)
        #print wav_arr[o,0,0],wav_arr[o,0,-1]
        print((wav_arr[o,0,lowi:uppi])[0],(wav_arr[o,0,lowi:uppi])[-1])
        for t in range(0,n_exp):
            bin_cnt[o,t,:],bin_err[o,t,:]=spectres.spectres(bin_ctr, wav_arr[o,t,lowi:uppi], cnt_arr[o,t,lowi:uppi], spec_errs=ptn_err[o,t,lowi:uppi])
    
        plt.figure(1,figsize=(15,4))
        plt.clf()
        plt.plot(wav_arr[o,10,:],cnt_arr[o,10,:]/np.nanmax(cnt_arr[o,10,:]),color='black')
        plt.plot(bin_ctr,bin_cnt[o,10,:]/np.nanmax(bin_cnt[o,10,:]),'.',markersize=10,color='red')
        plt.errorbar(bin_ctr,bin_cnt[o,10,:]/np.nanmax(bin_cnt[o,10,:]),
                     yerr=10*bin_err[o,10,:]/np.nanmax(bin_cnt[o,10,:]),fmt='None',ecolor='red')
        for b in bin_arr:
            plt.axvline(x=b,color='grey',linewidth=0.5)
        plt.ylim(-0.1,1.4)
        plt.xlim(3000,10000)
        plt.xlabel('Wavelength, [$\AA$]',fontsize=15)
        plt.ylabel('Relative Flux',fontsize=15)
        plt.figtext(0.15,0.80,'obj'+str(int(o)),fontsize=25,color='red')
        plt.show()
        plt.close()
        
    np.savez_compressed(SAVEPATH+'Binned_Data_'+str(int(width_bin))+'.npz',bins=bin_arr,bin_centers=bin_ctr,
                            bin_counts=bin_cnt,bin_err=bin_err,bin_ptn=bin_err)

