import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from datetime import datetime

from setup import *

def Gaussian(x,a,b,c,d):
    return a*np.exp(-((x-b)**2.)/(2.*c**2.))+d

def FlattenSpec():
    
    n_obj=int(np.load('SaveData/FinalMasks.npz')['masks'].shape[0])
    n_exp=np.load('SaveData/HeaderData.npz')['n_exp']
    
    flat_spec=np.empty([n_obj,n_exp,2*ypixels+ygap])*np.nan

    for i in range(0,n_obj):
        time0=datetime.now()
        print '-----------------'
        print '  OBJECT # ', i
        print '-----------------'
        obj_data=(np.load('SaveData/Corrected'+str(int(i))+'.npz'))['data']
        sub_bkgd=np.zeros_like(obj_data)*np.nan
        mask=(np.load('SaveData/FinalMasks.npz')['masks'])[i,:]
        y0=int(mask[1])
        n_rows=obj_data.shape[1]
        xwidth=obj_data.shape[2]
        xpix_ar=np.linspace(1,xwidth,xwidth)
        fwhm_ar=np.empty([n_exp,n_rows])
        fwhm_av=np.empty([n_exp])
        cent_ar=np.empty([n_exp,n_rows])
        for t in range(0,n_exp):
            if t%10==0:
                print '    -->> TIME: ',t
                print '       -- FITTING GAUSSIANS'
            frame=obj_data[t,:,:]
            for j in range(0,n_rows):
                row_data=frame[j,:]
                if not np.isfinite(row_data[0]):
                    continue
                bg_params=np.polyfit(np.append(xpix_ar[25:50],xpix_ar[xwidth-50:xwidth-25]),np.append(row_data[25:50],row_data[xwidth-50:xwidth-25]),1)
                background=(np.poly1d(bg_params))(xpix_ar)
                p0=np.array([np.nanmax(row_data),np.argmax(row_data),10,background[50]])
                try:
                    g_param,g_cov=curve_fit(Gaussian,xpix_ar,row_data,p0=p0,maxfev=10000)
                except RuntimeError:
                    sub_bkgd[t,j,:]=np.empty([len(row_data)])*np.nan
                else:
                    fwhm_ar[t,j]=2*np.sqrt(2*np.log(2))*g_param[2]
                    cent_ar[t,j]=int(g_param[1])
                    sub_bkgd[t,j,:]=row_data-background
                    #if t%10==0 and j%100==0:
                    #    plt.figure(1)
                    #    plt.clf()
                    #    plt.cla()
                    #    plt.plot(xpix_ar,row_data,color='black',linewidth=4.0)
                    #    plt.plot(xpix_ar,Gaussian(xpix_ar,*g_param),color='orange',linewidth=2.0,linestyle='-')
                    #    plt.plot(xpix_ar,background,color='blue',linewidth=2.0,linestyle='--')
                    #    plt.axvline(x=cent_ar[t,j],color='red',linewidth=1.5)
                    #    plt.axvline(x=cent_ar[t,j]-fwhm_ar[t,j], color='green',linestyle='--',linewidth=0.5)
                    #    plt.axvline(x=cent_ar[t,j]+fwhm_ar[t,j], color='green',linestyle='--',linewidth=0.5)
                    #    plt.axvline(x=cent_ar[t,j]-3.*fwhm_ar[t,j], color='darkgreen',linestyle='--',linewidth=1.5)
                    #    plt.axvline(x=cent_ar[t,j]+3.*fwhm_ar[t,j], color='darkgreen',linestyle='--',linewidth=1.5)
                    #    plt.figtext(0.1,0.9,str(int(i))+' '+str(int(t))+' '+str(int(j)))
                    #    plt.show(block=False)
            fwhm_av[t]=int(np.nanmedian(fwhm_ar[t,:]))
            if t%10==0:
                print '       -- SUMMING APERTURE'
            for j in range(0,n_rows):
                if not np.isfinite(row_data[0]):
                    continue
                low=np.nanmax([0,cent_ar[t,j]-3*fwhm_av[t]])
                up=np.nanmin([cent_ar[t,j]+3*fwhm_av[t],xwidth])
                #print flat_spec.shape, '     ', i,t,j+y0,j,n_rows
                flat_spec[i,t,j+y0]=np.sum(sub_bkgd[t,j,int(low):int(up)])
        plt.figure(2)
        plt.clf()
        plt.cla()
        for t in range(0,n_exp):
            if t%10==0:
                plt.plot(np.linspace(0,2*ypixels+ygap,2*ypixels+ygap),flat_spec[i,t,:])
        plt.figtext(0.2,0.8,'OBJECT '+str(int(i)),fontsize=15,color='red')
        plt.xlabel('Stitched Pixels')
        plt.ylabel('ADUs')
        plt.show(block=False)
        print ' '
        time1=datetime.now()
        print'          time to run: ', time1-time0
    np.savez('SaveData/FlattenedSpectra.npz',flat_spec=flat_spec,fwhm_ar=fwhm_ar,fwhm_av=fwhm_av,cent_ar=cent_ar)
    return flat_spec
        
    
