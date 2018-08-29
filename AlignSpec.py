import numpy as np
np.seterr('ignore')

import scipy
from scipy.signal import convolve
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import medfilt

import matplotlib
import matplotlib.pyplot as plt

from outlier_removal import outlierr_c
from bin_y_shift import bin_y_shift

from setup import *




def func_gaus(x,sigma):
    return 1.0-np.exp(-(1./2.)*(x/(sigma))**2.)

def AlignSpec(gris,osr,fwhm_s,fwhm_t,ks,olv,wavelength_path,obj_name,SAVEPATH,
              ex,binny,ver,ver_l,time_trim,skip):
    masks=np.load(SAVEPATH+'FinalMasks.npz')['masks']

    input_data=np.load(SAVEPATH+'FlattenedSpectra.npz')['flat_spec']
    n_obj=input_data.shape[0]
    n_exp=input_data.shape[1]
    n_pix=input_data.shape[2]
    n_pix_chip=(n_pix-ygap)/2
    
    cnv_data=np.empty([n_obj,n_exp,int(osr*n_pix)])*np.nan
    ovs_data=np.empty([n_obj,n_exp,int(osr*n_pix)])*np.nan
    int_data=np.empty([n_obj,n_exp,n_pix])*np.nan

    pix_ar=np.linspace(0,n_pix-1,n_pix)
    pix_ar_os=np.linspace(0,n_pix-1,int(osr*n_pix))
    y_orig=np.copy(pix_ar)
    y_orig_os=np.copy(pix_ar_os)
#     print pix_ar
#     print (n_pix-ygap)/2, n_pix
#     print ypixels, 2*ypixels+ygap
#     print 
    if binny>1:
        pix_ar=bin_y_shift(pix_ar,binny)
        pix_ar_os=bin_y_shift(pix_ar_os,binny)

       
    print pix_ar
    pix_ar=np.flip(pix_ar,axis=0)
    pix_ar_os=np.flip(pix_ar_os,axis=0)
    print pix_ar
    
    shift_pixels=np.empty([n_obj,n_exp,n_pix])*np.nan
    wav_ar=np.empty([n_obj,n_exp,n_pix])*np.nan
    y_shift=np.empty([n_obj,n_exp])*np.nan
    
    #medfilt_data=medfilt(input_data,kernel_size=[1,1,window])
    #stddev_data=np.empty([n_obj,n_exp,n_pix])
    #for p in range(window,n_pix-window):
     #   stddev_data[:,:,p]=np.std(input_data[:,:,np.int(p-(window/2)-1):np.int(p+(window/2)-1)])
    #for t in range(2,n_exp-2):
    #    stddev_data[:,t,:]=np.nanstd(input_data[:,t-2:t+2,:])
        
    #smooth_data=input_data
    #del input_data
    
    #spectral outlier/smoothing function
    for o in range(0,n_obj):
        print '-----------------'
        print '  OBJECT # ', o
        print '-----------------'
        if o in skip:
            print '--------- BAD WAVELENGTH SOLUTION'  
            continue
        
        #if o==1 or o==2 or o==5 or o==7:
        #    continue
        counter=0
        
        norm=matplotlib.colors.Normalize(vmin=0,vmax=n_exp)
        colors=matplotlib.cm.viridis
        scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
        scal_m.set_array([])

        fig=plt.figure(201,figsize=(15,2))
        for t in range(0,n_exp):
            plt.plot(pix_ar,input_data[o,t,:],color=scal_m.to_rgba(t),linewidth=1.0)
        # [left, bottom, width, height
        plt.xlabel('Pixels')
        cbaxes = fig.add_axes([0.15, 0.2, 0.02, 0.6]) 
        cb = plt.colorbar(scal_m, cax = cbaxes)  
        plt.show(block=False)
        plt.close()
        
        print ' --Filtering...'
        tc=0
        for t in range(0,n_exp-time_trim):
            counter,input_data[o,t,:]=outlierr_c(np.copy(input_data[o,t,:]),ks,olv)
            tc+=counter
            counter,input_data[o,t,:]=outlierr_c(np.copy(input_data[o,t,:]),ks,olv)
            tc+=counter

        print '       -->>',float(tc)/float(n_exp), tc
        

        fig=plt.figure(102,figsize=(15,2))
        for t in range(0,n_exp):
            plt.plot(pix_ar,input_data[o,t,:],color=scal_m.to_rgba(t),linewidth=1.0)
        # [left, bottom, width, height
        plt.xlabel('Pixels')
        cbaxes = fig.add_axes([0.15, 0.2, 0.02, 0.6]) 
        cb = plt.colorbar(scal_m, cax = cbaxes)  
        plt.figtext(0.2,0.7,tc,color='black',fontsize=25)
        plt.show(block=False)
        plt.close()
        
        print ' --Oversampling...'
        for t in range(0,n_exp-time_trim):
            interp_d=interp1d(pix_ar,input_data[o,t,:],bounds_error=False,fill_value=np.nan)
            ovs_data[o,t,:]=interp_d(pix_ar_os)
        fig=plt.figure(201,figsize=(15,2))
        for t in range(0,n_exp):
            if t%10==0:
                plt.plot(pix_ar_os,ovs_data[o,t,:],color=scal_m.to_rgba(t),linewidth=1.0)
        # [left, bottom, width, height
        plt.xlabel('Pixels')
        cbaxes = fig.add_axes([0.15, 0.2, 0.02, 0.6]) 
        cb = plt.colorbar(scal_m, cax = cbaxes)  
        plt.show(block=False)
        plt.close()
       
        
        print ' --Convolving with Gaussian...'
        if fwhm_t==False:
            fwhm=fwhm_s*osr
            sigma=fwhm/2.355
            width=2.*fwhm
            width_line=np.linspace(-width/2,width/2.,width)
            gaus_cnv=func_gaus(width_line,sigma)
            for t in range(0,n_exp-time_trim):
                cnv_data[o,t,:]=np.convolve(np.nan_to_num(ovs_data[o,t,:]),gaus_cnv,mode='same')
            
        else:
            fwhm_arr=(np.load(SAVEPATH+'FlattenedSpectra.npz')['gaus_params'])[o,:,:,2]
            fwhm_arr=(2.*np.sqrt(2.*np.log(2.))*np.nanmedian(fwhm_arr,axis=1))
            #fwhm=fwhm_arr[o,t]
            print fwhm_arr
            
            for t in range(0,n_exp-time_trim):
                fwhm=fwhm_arr[t]*osr
                if np.isfinite(fwhm)==False:
                    fwhm=fwhm_s*osr
                sigma=fwhm/2.355
                width=2.*fwhm
                width_line=np.linspace(-width/2,width/2.,width)
                gaus_cnv=func_gaus(width_line,sigma)
#                 plt.figure(202,figsize=(6,6))
#                 plt.plot(width_line,gaus_cnv)
#                 plt.show()
                cnv_data[o,t,:]=np.convolve(np.nan_to_num(ovs_data[o,t,:]),gaus_cnv,mode='same')
                
        
        fig=plt.figure(103,figsize=(15,2))
        for t in range(0,n_exp):
            if t%10==0:
                plt.plot(pix_ar_os,cnv_data[o,t,:],color=scal_m.to_rgba(t),linewidth=1.0)
        # [left, bottom, width, height
        plt.xlabel('Pixels')
        cbaxes = fig.add_axes([0.15, 0.2, 0.02, 0.6]) 
        cb = plt.colorbar(scal_m, cax = cbaxes)  
        plt.show(block=False)
        plt.close()
        

        print ' --Cross Correlating in Time...'
        
#         ovs_data=np.flip(ovs_data,axis=2)
#         cnv_data=np.flip(cnv_data,axis=2)
#         input_data=np.flip(input_data,axis=2)
        
        # first, apply wavelength solution to first point in time to identify pixel locations of major atmospheric lines
        filew=wavelength_path+'Cal_'+str(int(o))+'_out.txt'
       
        if gris==150:
            coeff=np.genfromtxt(filew,skip_header=4,skip_footer=14,usecols=[1])
            new_pix=np.genfromtxt(filew,skip_header=4+coeff.size+3,usecols=[1])
            cor_wav=np.genfromtxt(filew,skip_header=4+coeff.size+3,usecols=[2])
        else:
            coeff=np.genfromtxt(filew,skip_header=4,skip_footer=25,usecols=[1])
            new_pix=np.genfromtxt(filew,skip_header=4+coeff.size+3,usecols=[1])
            cor_wav=np.genfromtxt(filew,skip_header=4+coeff.size+3,usecols=[2])
        
        print coeff
        print new_pix
        print cor_wav
        
            
        order=len(coeff)-1
        
        ALL_PIXELS=np.empty([n_obj,len(new_pix)])
            
        ALL_PIXELS[o,:]=new_pix#-(y0_fflip-y0_o)
        wav_func=np.poly1d(np.polyfit(ALL_PIXELS[o,:],cor_wav,order))
        
        wave_first=wav_func(pix_ar_os)
        
#         plt.figure(201,figsize=(10,10))
#         plt.plot(pix_ar_os,wave_first,color='red')
#         plt.axvline(x=4100)
#         plt.axhline(y=7600)
#         plt.show(block=False)
#         print wave_first.shape
        
#         wave_first=np.flip(wave_first,axis=0)
#         ovs_data=np.flip(ovs_data,axis=2)
#         cnv_data=np.flip(cnv_data,axis=2)
#         input_data=np.flip(input_data,axis=2)


        #identifying line locations in first exposure
        #line_waves=np.array([7615,6867,6563,5896],dtype='float')
        line_waves=np.array([6563,5896],dtype='float')
        search_win=30/binny
        
        line_ind_low=np.empty([len(line_waves)],dtype=int)*np.nan
        line_ind_upp=np.empty([len(line_waves)],dtype=int)*np.nan
        line_arrs={}
        line_ind_ac0=np.empty([len(line_waves)],dtype=int)*np.nan
        
        line_mins=np.empty([n_exp,len(line_waves)],dtype=int)*np.nan
        
        for l in range(0,len(line_ind_low)): 
            w=line_waves[l]
            upp=int(np.argmin(np.abs(wave_first-(w-search_win))))
            low=int(np.argmin(np.abs(wave_first-(w+search_win))))
            test=cnv_data[o,0,low:upp]
            tests=wave_first[low:upp]
            w=tests[np.argmin(test)]
            print line_waves[l], w
            line_waves[l]=w
            

        for l in range(0,len(line_ind_low)):
            w=line_waves[l]
            line_ind_upp[l]=int(np.argmin(np.abs(wave_first-(w-search_win))))
            line_ind_low[l]=int(np.argmin(np.abs(wave_first-(w+search_win))))
            print w,':', line_ind_low[l], line_ind_upp[l]
            line_arrs[str(int(l))]=cnv_data[o,0,int(line_ind_low[l]):int(line_ind_upp[l])]
            line_ind_ac0[l]=np.argmin(line_arrs[str(int(l))])
        for t in range(0,n_exp-time_trim): 
            if ver==True:
                if t%10==0:
                    print 'TIME:', t
            for l in range(0,len(line_ind_low)):
                ref=cnv_data[o,0,int(line_ind_low[l]):int(line_ind_upp[l])]
                line_data=cnv_data[o,t,int(line_ind_low[l]):int(line_ind_upp[l])]
                #ks=int(fwhm_s*osr)
                #if ks%2==0:
                #    ks+=1
                #ref=medfilt(ref,kernel_size=ks)
                #line_data=medfilt(line_data,kernel_size=ks)
                ref_n=np.pad((ref/np.nanmax(ref)),(len(ref),len(ref)),'edge')
                lda_n=np.pad((line_data/np.nanmax(line_data)),(len(line_data),len(line_data)),'edge')
                ref_n-=ref_n[0]
                lda_n-=lda_n[0]
                npcorr=np.correlate(ref_n, lda_n,'same')
                s_t_l=np.argmax(npcorr)-(len(ref_n)/2.)
                if ver==True:
                    if t%10==0:
                        fig,ax=plt.subplots(1,2,figsize=(15,4))
                        fig.subplots_adjust(wspace=0.0,hspace=0.0)
                        ax[0].plot(npcorr,color='black',linewidth=3.0)
                        ax[0].axvline(x=(len(ref_n)/2.),color='grey',linewidth=2.0)
                        ax[0].axvline(x=np.argmax(npcorr),color='red',linewidth=1.0)
                        ax[1].plot(ref_n,color='black',linewidth=4.0)
                        ax[1].plot(lda_n,color='red',linewidth=2.0)
                        ax[1].axvline(x=np.argmin(ref_n),color='grey',linewidth=2.0)
                        ax[1].axvline(x=np.argmin(lda_n),color ='tomato',linewidth=1.0)
                        plt.figtext(0.2,0.8,str(t),fontsize=25)
                        plt.show(block=False)
                        print '  --> line', line_waves[l],' shift: ', s_t_l
                line_mins[t,l]=s_t_l
                
                   
#         #identifying line locations in first exposure
#         o2_7594_ind_upp=np.where(np.abs(wave_first-(7594-30))==np.nanmin(np.abs(wave_first-(7594-30))))[0][0]
#         o2_7594_ind_low=np.where(np.abs(wave_first-(7594+30))==np.nanmin(np.abs(wave_first-(7594+30))))[0][0]
#         print o2_7594_ind_low,o2_7594_ind_upp
#         o2_7594_ar_0=ovs_data[o,0,o2_7594_ind_low:o2_7594_ind_upp]
#         #o2_7594_ac_0=np.where(ovs_data[o,0,:]==np.nanmin(o2_7594_ar_0))[0][0]
#         o2_7594_ac_0=np.argmin(o2_7594_ar_0)
        
        
#         o2_6867_ind_upp=np.where(np.abs(wave_first-(6867-30))==np.nanmin(np.abs(wave_first-(6867-30))))[0][0]
#         o2_6867_ind_low=np.where(np.abs(wave_first-(6867+30))==np.nanmin(np.abs(wave_first-(6867+30))))[0][0]
#         print o2_6867_ind_low,o2_6867_ind_upp
#         o2_6867_ar_0=ovs_data[o,0,o2_6867_ind_low:o2_6867_ind_upp]
#         #o2_6867_ac_0=np.where(ovs_data[o,0,:]==np.nanmin(o2_6867_ar_0))[0][0]
#         o2_6867_ac_0=np.argmin(o2_6867_ar_0)
        
#         ha_6563_ind_upp=np.where(np.abs(wave_first-(6563-30))==np.nanmin(np.abs(wave_first-(6563-30))))[0][0]
#         ha_6563_ind_low=np.where(np.abs(wave_first-(6563+30))==np.nanmin(np.abs(wave_first-(6563+30))))[0][0]
#         print ha_6563_ind_low,ha_6563_ind_upp
#         ha_6563_ar_0=ovs_data[o,0,ha_6563_ind_low:ha_6563_ind_upp]
#         #ha_6563_ac_0=np.where(ovs_data[o,0,:]==np.nanmin(ha_6563_ar_0))[0][0]
#         ha_6563_ac_0=np.argmin(ha_6563_ar_0)
        
#         na_5896_ind_upp=np.where(np.abs(wave_first-(5896-30))==np.nanmin(np.abs(wave_first-(5896-30))))[0][0]
#         na_5896_ind_low=np.where(np.abs(wave_first-(5896+30))==np.nanmin(np.abs(wave_first-(5896+30))))[0][0]
#         print na_5896_ind_low,na_5896_ind_upp
#         na_5896_ar_0=ovs_data[o,0,na_5896_ind_low:na_5896_ind_upp]
#         #na_5896_ac_0=np.where(ovs_data[o,0,:]==np.nanmin(na_5896_ar_0))[0][0]
#         na_5896_ac_0=np.argmin(na_5896_ar_0)
        
#         for t in range(0,n_exp-time_trim):
#             #identify line locations in each exposure
#             o2_7594_ar=ovs_data[o,t,o2_7594_ind_low:o2_7594_ind_upp]
#             #o2_7594_ac=np.where(ovs_data[o,t,:]==np.nanmin(o2_7594_ar))[0][0]
#             o2_7594_ac=np.argmin(o2_7594_ar)
        
#             o2_6867_ar=ovs_data[o,t,o2_6867_ind_low:o2_6867_ind_upp]
#             #o2_6867_ac=np.where(ovs_data[o,t,:]==np.nanmin(o2_6867_ar))[0][0]
#             o2_6867_ac=np.argmin(o2_6867_ar)
        
#             ha_6563_ar=ovs_data[o,t,ha_6563_ind_low:ha_6563_ind_upp]
#             #ha_6563_ac=np.where(ovs_data[o,t,:]==np.nanmin(ha_6563_ar))[0][0]
#             ha_6563_ac=np.argmin(ha_6563_ar)
        
#             na_5896_ar=ovs_data[o,t,na_5896_ind_low:na_5896_ind_upp]
#             #na_5896_ac=np.where(ovs_data[o,t,:]==np.nanmin(na_5896_ar))[0][0]
#             na_5896_ac=np.argmin(na_5896_ar)
       
            #calculating line shifts
        
#             shift_o2_7594=o2_7594_ac_0-o2_7594_ac
#             shift_o2_6867=o2_6867_ac_0-o2_6867_ac
#             shift_ha_6563=ha_6563_ac_0-ha_6563_ac
#             shift_na_5896=na_5896_ac_0-na_5896_ac
            
#            shift=np.nanmedian([shift_o2_7594,shift_o2_6867,shift_ha_6563,shift_na_5896])
            shift=np.nanmedian(line_mins[t,:])
            y_shift[o,t]=float(shift)/float(osr)
            
            
            if t%10==0:
                if ver_l==True:
                    print ' '
                    print '*****************'
                    print 'TIME: ', t
                    print 'o2_7594 shift: NOT USED'#, line_mins[t,0]
                    print 'o2_6867 shift: NOT USED'#, line_mins[t,1]
                    print 'ha_6563 shift: ', line_mins[t,0]
                    print 'na_5896 shift: ', line_mins[t,1]

                    print 'mean shift: ', shift, float(shift)/float(osr), y_shift[o,t]
                
               
                if ver==True:
                    plt_ex=10
                    fig,ax=plt.subplots(1,4,figsize=(12,3))
                    
                    for l in range(0,len(line_ind_low)):
                        plt_data=cnv_data[o,:,int(line_ind_low[l]-plt_ex):int(line_ind_upp[l]+plt_ex+1)]
                        wave_dat=wave_first[int(line_ind_low[l]-plt_ex):int(line_ind_upp[l]+plt_ex+1)]
                        ax[l].plot(wave_dat,plt_data[0,:]/np.nanmax(plt_data[0,:]), color='black', linewidth=3.0)
                        ax[l].plot(wave_dat,plt_data[t,:]/np.nanmax(plt_data[t,:]), color='red', linewidth=1.0)
                        ax[l].axvline(x=wave_dat[int(line_ind_ac0[l]+plt_ex)], linestyle='--',color='grey',linewidth=1.5)
                        ax[l].axvline(x=wave_dat[int(line_ind_ac0[l]+plt_ex-line_mins[t,l])], 
                                      linestyle='--',color='salmon',linewidth=0.5)
                        ax[l].set_title(str(int(line_waves[l])))
                                       
            
#                     plt_o2_7594=ovs_data[o,:,o2_7594_ind_low-plt_ex:o2_7594_ind_upp+plt_ex]
#                     wav_7594=wave_first[o2_7594_ind_low:o2_7594_ind_upp]
                
#                     plt_o2_6867=ovs_data[o,:,o2_6867_ind_low-plt_ex:o2_6867_ind_upp+plt_ex]
#                     wav_6867=wave_first[o2_6867_ind_low:o2_6867_ind_upp]
                
#                     plt_ha_6563=ovs_data[o,:,ha_6563_ind_low-plt_ex:ha_6563_ind_upp+plt_ex]
#                     wav_6563=wave_first[ha_6563_ind_low:ha_6563_ind_upp]
                
#                     plt_na_5896=ovs_data[o,:,na_5896_ind_low-plt_ex:na_5896_ind_upp+plt_ex]
#                     wav_5896=wave_first[na_5896_ind_low:na_5896_ind_upp]
                
#                     fig,ax=plt.subplots(1,4,figsize=(12,3))
                
#                     ax[0].plot(wave_first[o2_7594_ind_low-plt_ex:o2_7594_ind_upp+plt_ex],
#                              plt_o2_7594[0,:]/np.nanmax(plt_o2_7594[0,:]), color='black', linewidth=3.0)
#                     ax[0].plot(wave_first[o2_7594_ind_low-plt_ex:o2_7594_ind_upp+plt_ex],
#                              plt_o2_7594[t,:]/np.nanmax(plt_o2_7594[t,:]), color='red', linewidth=1.0)
#                     ax[0].axvline(x=wav_7594[o2_7594_ac_0], linestyle='--',color='grey',linewidth=1.5)
#                     ax[0].axvline(x=wav_7594[o2_7594_ac], linestyle='--',color='salmon',linewidth=0.5)
#                     ax[0].set_title('7594 O$_{2}$')
                
#                     ax[1].plot(wave_first[o2_6867_ind_low-plt_ex:o2_6867_ind_upp+plt_ex],
#                              plt_o2_6867[0,:]/np.nanmax(plt_o2_6867[0,:]), color='black', linewidth=3.0)
#                     ax[1].plot(wave_first[o2_6867_ind_low-plt_ex:o2_6867_ind_upp+plt_ex],
#                              plt_o2_6867[t,:]/np.nanmax(plt_o2_6867[t,:]), color='red', linewidth=1.0)
#                     ax[1].axvline(x=wav_6867[o2_6867_ac_0], linestyle='--',color='grey',linewidth=1.5)
#                     ax[1].axvline(x=wav_6867[o2_6867_ac], linestyle='--',color='salmon',linewidth=0.5)
#                     ax[1].set_title('6867 O$_{2}$')
                
#                     ax[2].plot(wave_first[ha_6563_ind_low-plt_ex:ha_6563_ind_upp+plt_ex],
#                              plt_ha_6563[0,:]/np.nanmax(plt_ha_6563[0,:]), color='black', linewidth=3.0)
#                     ax[2].plot(wave_first[ha_6563_ind_low-plt_ex:ha_6563_ind_upp+plt_ex],
#                              plt_ha_6563[t,:]/np.nanmax(plt_ha_6563[t,:]), color='red', linewidth=1.0)
#                     ax[2].axvline(x=wav_6563[ha_6563_ac_0], linestyle='--',color='grey',linewidth=1.5)
#                     ax[2].axvline(x=wav_6563[ha_6563_ac], linestyle='--',color='salmon',linewidth=0.5)
#                     ax[2].set_title('6563 H$\\alpha$')
                
#                     ax[3].plot(wave_first[na_5896_ind_low-plt_ex:na_5896_ind_upp+plt_ex],
#                              plt_na_5896[0,:]/np.nanmax(plt_na_5896[0,:]), color='black', linewidth=3.0)
#                     ax[3].plot(wave_first[na_5896_ind_low-plt_ex:na_5896_ind_upp+plt_ex],
#                              plt_na_5896[t,:]/np.nanmax(plt_na_5896[t,:]), color='red', linewidth=1.0)
#                     ax[3].axvline(x=wav_5896[na_5896_ac_0], linestyle='--',color='grey',linewidth=1.5)
#                     ax[3].axvline(x=wav_5896[na_5896_ac], linestyle='--',color='salmon',linewidth=0.5)
#                     ax[3].set_title('5896 Na')
                
                    plt.show(block=False)
                
                
            
            shift_pixels[o,t,:]=pix_ar+float(shift)/float(osr)
            
        wav_ar[o,:,:]=wav_func(shift_pixels[o,:,:])
        
        print y_shift[o,:]
        plt.figure(104,figsize=(15,2))
        plt.plot(np.linspace(1,n_exp,n_exp),y_shift[o,:],'.',markersize=12,markerfacecolor=scal_m.to_rgba(2),
                markeredgecolor='black')
        plt.ylim(1.5*np.nanmin(y_shift[o,:]),1.5*np.nanmax(y_shift[o,:]))
        plt.xlabel('Exposure Number',fontsize=15)
        plt.show(block=False)
        
        
        plt.figure(105,figsize=(15,2))
        
        plt.plot(wav_ar[o,0,:],input_data[o,0,:]/np.nanmax(input_data[o,0,:]),color='black',linewidth=2.0)
        plt.plot(wave_first,cnv_data[o,0,:]/np.nanmax(cnv_data[o,0,:]),color='red',linewidth=1.0)
        plt.axvline(x=7593.7,color='grey',linewidth=0.5,linestyle='--')
        plt.axvline(x=6867.19,color='grey',linewidth=0.5,linestyle='--')
        plt.axvline(x=6562.81,color='grey',linewidth=0.5,linestyle='--')
        plt.axvline(x=5895.9,color='grey',linewidth=0.5,linestyle='--')
        plt.axvline(x=5889.9,color='grey',linewidth=0.5,linestyle='--')
        plt.xlabel('Wavelength, [A]',fontsize=15)
        plt.show(block=False)
    
    np.savez_compressed(SAVEPATH+'ShiftedSpec_All.npz',
                            data=input_data,pixels=shift_pixels,wave=wav_ar,yshift=y_shift)
