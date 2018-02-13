import numpy as np
np.seterr('ignore')

import scipy
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsec

#from setup import *
def BinFunc(data,wave,start,end,width):
    #cnt_arr=np.load(DATAFILE)['data']
    #wav_arr=np.load(DATAFILE)['wave']
    
    cnt_arr=data
    wav_arr=wave
    
    #cnt_arr=np.flip(cnt_arr,axis=2)
    #wav_arr=np.flip(wav_arr,axis=2)
    
    #n_obj=cnt_arr.shape[0]
    #n_exp=cnt_arr.shape[1]
    
    pixels=len(cnt_arr)
    pix_arr=np.linspace(0,len(cnt_arr),len(cnt_arr))
    
    width_bin=width
    numbins=int((end-start)/width_bin)
    bin_arr=np.linspace(start,end,numbins+1)
    bin_ctr=np.array([])
    for b in range(0,len(bin_arr)-1):
        bin_ctr=np.append(bin_ctr,bin_arr[b]+width_bin/2.)
    
    print( '  -->> From Lambda=', start, ' to Lambda=', end)
    print( '  -->> TOTAL OF ', numbins, 'WAVELENGTH BINS')
    print( '       Bin Centers: ', bin_ctr)
    print( '       Bin Array:   ', bin_arr)
    print( '       Bin Width:   ', width_bin)
    
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_arr),vmax=np.max(bin_arr))
    #colors=matplotlib.cm.RdYlBu_r
    #colors=matplotlib.cm.Spectral_r
    colors=matplotlib.cm.jet
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])
    
    bin_cnt=np.empty([numbins])*np.nan
    #bin_err_up=np.empty([numbins])*np.nan
    #bin_err_dn=np.empty([numbins])*np.nan
    #bin_err_pt=np.empty([numbins])*np.nan

    #bin_err=np.empty([numbins,n_obj])*np.nan
    #bin_ptn=np.empty([n_exp,numbins,n_obj])*np.nan
    
   
            #if t%10==0:
            #    print '          -> TIME', t
    bine=0
            #err1=0
            #err2=0
    wave_cntr=start
            #print wave_cntr
    p=0
    while np.isfinite(wav_arr[p])==False and p<pixels-2:
        p+=1
    p+=1
    while wav_arr[p]>wave_cntr+width_bin:
                #print p,wav_arr[s,t,p]
        wave_cntr+=width_bin
    while wave_cntr<bin_arr[-1] and p < pixels-1:
        counts=0
        counter=0
                #if t==0:
                #    print bine, counts, np.where(bin_arr==wave_cntr)
        if p>0:
            while (wav_arr[p-1]+wav_arr[p])/2. < wave_cntr+width_bin and p <pixels-1:
                lowerbound=(wav_arr[p-1]+wav_arr[p])/2.
                upperbound=(wav_arr[p+1]+wav_arr[p])/2.
                if wave_cntr>lowerbound and wave_cntr<upperbound:
                    percent=1.0-(wave_cntr-lowerbound)/(upperbound-lowerbound)
                    counts+=np.nan_to_num(percent*cnt_arr[p])*(upperbound-lowerbound)
                    #print 'CASE A', bine, p, wav_arr[p], wave_cntr,counts,'..',cnt_arr[p],percent,lowerbound,upperbound
                            #err1+=np.nan_to_num(((percent*err_up[t-1,p,s]))**2.)
                            #err2+=np.nan_to_num(((percent*err_dn[t-1,p,s]))**2.)
                            #ptne+=np.nan_to_num(((percent*ptn_err[t-1,p,s]))**2.)               
                if wave_cntr<lowerbound and wave_cntr+width_bin>upperbound:
                    counts+=np.nan_to_num(cnt_arr[p])*(upperbound-lowerbound)
                    #print 'CASE B', bine, p, wav_arr[p], wave_cntr,counts,'..',cnt_arr[p]
                            #err1+=np.nan_to_num(((err_up[t-1,p,s]))**2.)
                            #err2+=np.nan_to_num(((err_dn[t-1,p,s]))**2.)
                            #ptne+=np.nan_to_num(((ptn_err[t-1,p,s]))**2.)
                if wave_cntr+width_bin>lowerbound and wave_cntr+width_bin<upperbound:
                    percent=(wave_cntr+width_bin-lowerbound)/(upperbound-lowerbound)
                    counts+=np.nan_to_num(percent*cnt_arr[p])*(upperbound-lowerbound)
                    #print 'CASE C', bine, p, wav_arr[p], wave_cntr,counts,'..',cnt_arr[p],percent,lowerbound,upperbound
                            #err1+=np.nan_to_num(((percent*err_up[t-1,p,s]))**2.)
                            #err2+=np.nan_to_num(((percent*err_dn[t-1,p,s]))**2.)
                            #ptne+=np.nan_to_num(((percent*ptn_err[t-1,p,s]))**2.)
                counter=counter+1
                p+=1  # goes to next pixel in this bin
        if p < pixels-1:
            p-=1   #goes down pixel value to add second part of pixel to next bin
                #if t==0:
                #    print bine, counts, np.where(bin_arr==wave_cntr)
        bin_cnt[bine]=counts/counter
                #bin_err[t-1,bin,s]=np.nanmean([np.sqrt(err1+ptne),np.sqrt(err2+ptne)])
                #bin_ptn[t-1,bin,s]=np.sqrt(ptne)
        wave_cntr+=width_bin
        bine+=1
        #print '--------------------------'
                #if t%10==0:
                #    print '             ->', counts
    plt.clf()
    plt.figure(1,figsize=((end-start)/400,4.))
    plt.plot(wav_arr,cnt_arr/np.nanmax(cnt_arr),color='black')
    plt.plot(bin_ctr,bin_cnt/np.nanmax(bin_cnt),'.',markersize=10,color='red')
       # plt.errorbar(bin_ctr,bin_cnt[0,:,s]/np.nanmax(bin_cnt[0,:,s]),yerr=10*bin_err[0,:,s]/np.nanmax(bin_cnt[0,:,s]),fmt=None,ecolor='red')
    for b in bin_arr:
        plt.axvline(x=b,color='grey',linewidth=0.5)
    plt.ylim(-0.1,1.4)
    plt.xlim(3000,10000)
    plt.xlabel('Wavelength, [$\AA$]',fontsize=15)
    plt.ylabel('Relative Flux',fontsize=15)
    #plt.figtext(0.15,0.80,'obj'+str(int(s)),fontsize=25,color='red')
    plt.show(block=False)
    #plt.pause(2.0)
    plt.close()
    #np.savez_compressed(SAVEPATH+'Binned_Data.npz',bins=bin_arr,bin_centers=bin_ctr,bin_counts=bin_cnt)
    return bin_ctr,bin_cnt
    
