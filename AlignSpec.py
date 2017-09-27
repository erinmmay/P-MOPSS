import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import medfilt

import matplotlib.pyplot as plt


def func_gaus(x,sigma):
    return 1.0-np.exp(-(1./2.)*(x/(sigma))**2.)

def AlignSpec(osr,window,fwhm,wavelength_path,obj_name,SAVEPATH):
    input_data=np.load(SAVEPATH+'FlattenedSpectra.npz')['flat_spec']
    n_obj=input_data.shape[0]
    n_exp=input_data.shape[1]
    n_pix=input_data.shape[2]
    
    cnv_data=np.empty([n_obj,n_exp,n_pix])
    ovs_data=np.empty([n_obj,n_exp,osr*n_pix])
    int_data=np.empty([n_obj,n_exp,n_pix])
    
    pix_ar=np.linspace(n_pix,0,n_pix)
    pix_ar_os=np.linspace(n_pix,0,osr*n_pix)
    
    shift_pixels=np.empty([n_obj,n_exp,n_pix])
    
    wav_ar=np.empty([n_obj,n_exp,n_pix])
    
    #medfilt_data=medfilt(input_data,kernel_size=[1,1,window])
    #stddev_data=np.empty([n_obj,n_exp,n_pix])
    #for p in range(window,n_pix-window):
     #   stddev_data[:,:,p]=np.std(input_data[:,:,np.int(p-(window/2)-1):np.int(p+(window/2)-1)])
    #for t in range(2,n_exp-2):
    #    stddev_data[:,t,:]=np.nanstd(input_data[:,t-2:t+2,:])
        
    smooth_data=input_data
    #del input_data
    
    #spectral outlier/smoothing function
    for o in range(0,n_obj):
        print '-----------------'
        print '  OBJECT # ', o
        print '-----------------'
        if o==1 or o==2 or o==5 or o==7:
            continue
        counter=0
        print ' --Filtering...'
        for t in range(0,n_exp):
            if t%10==0:
                print '    -->> TIME: ',t
            for p in range(0,n_pix):
                minp=np.max([0,p-window])
                maxp=np.min([p+window,n_pix])
                median=np.nanmedian(np.append(input_data[o,t,minp:p],input_data[o,t,p+1:maxp]))
                stdev=np.nanstd(np.append(input_data[o,t,minp:p],input_data[o,t,p+1:maxp]))
                if np.abs(input_data[o,t,p]-median)>3.*stdev:
                    smooth_data[o,t,p]=np.nanmedian(input_data[o,t-2:t+2,p])
                    counter+=1
        print '       -->>',counter
        
        print ' --Convolving with Gaussian...'
        sigma=fwhm/2.355
        width=2.*fwhm
        width_line=np.linspace(-width/2,width*2.,width)
        gaus_cnv=func_gaus(width_line,sigma)
        for t in range(0,n_exp):
            cnv_data[o,t,:]=convolve(smooth_data[o,t,:],gaus_cnv,'same')
        
        print ' --Oversampling...'
        for t in range(0,n_exp):
            interp_d=interp1d(pix_ar,cnv_data[o,t,:],bounds_error=False,fill_value=np.nan)
            ovs_data[o,t,:]=interp_d(pix_ar_os)
       
        print ' --Cross Correlating in Time...'
        time0=np.nan_to_num(ovs_data[o,0,:]/np.nanmax(ovs_data[o,0,:]))
        for t in range(0,n_exp):
            comp=np.nan_to_num(ovs_data[o,t,:]/np.nanmax(ovs_data[o,t,:]))
            pix_shift=np.argmax(np.correlate(time0,comp,'full'))-(len(pix_ar_os)-1)
            if t%10==0:
                print '    -->> TIME: ',t,'    pixel shift: ',float(pix_shift)/float(osr)
            shift_pixels[o,t,:]=pix_ar+float(pix_shift)/float(osr)
        
        #### wavelength solution ####
        if o==0:
            filew=wavelength_path+obj_name+'_out.txt'
            #coeff=np.genfromtxt(wavelength_path+obj_name+'_out.txt',skip_header=4,skip_footer=25,usecols=[1])
    
        else:
            filew=wavelength_path+'Cal'+str(int(o))+'_out.txt'
        coeff=np.genfromtxt(filew,skip_header=4,skip_footer=25,usecols=[1])
        new_pix=np.genfromtxt(filew,skip_header=4+coeff.size+3,usecols=[1])
        cor_wav=np.genfromtxt(filew,skip_header=4+coeff.size+3,usecols=[2])
        wav_func=np.poly1d(np.polyfit(new_pix,cor_wav,1))
        wav_ar[o,:,:]=wav_func(shift_pixels[o,:,:])
        
        #interperolated data
        #for t in range(0,n_exp):
        #    inter=interp1d(wav_ar[o,t,:],smooth_data[o,t,:])
        #    int_data[o,t,:]=interp1d(wav_ar[o,t,:],smooth_data[o,t,:])
        
        plt.plot(wav_ar[o,0,:],smooth_data[o,0,:]/np.nanmax(smooth_data[o,0,:]),color='black',linewidth=2.0)
        plt.plot(wav_ar[o,0,:],cnv_data[o,0,:]/np.nanmax(cnv_data[o,0,:]),color='red',linewidth=1.0)
        plt.show()
    
    np.savez(SAVEPATH+'ShiftedSpec_All.npz',data=smooth_data,convolved=cnv_data,pixels=shift_pixels,wave=wav_ar)
