import numpy as np
import batman

import matplotlib
from matplotlib import font_manager as fm, rcParams
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#%matplotlib inline

params = {'font.family' : 'serif'}
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
matplotlib.rcParams.update(params)

def lnprior(theta,par_n,par_d,par_u):
    # par_c=theta
    # # print(par_n)
    # # print(par_c)
    # # print(all(np.greater(par_c,par_d)))
    # # print(all(np.less(par_c,par_u)))
    # if 'c1' in par_n:  #if fitting for limb darkening too, apply appropriate bounds
    #     ind1=np.where(par_n=='c1')[0]
    #     ind2=np.where(par_n=='c2')[0]
    #     #print(ind,par_n[ind],par_n[ind+1])
    #     # if all(np.greater(par_c,par_d)) and all(np.less(par_c,par_u)) and par_c[ind1]>par_c[ind2] and par_c[ind1]+par_c[ind2]<1.0:
    #     if all(np.greater(par_c,par_d)) and all(np.less(par_c,par_u)) and par_c[ind1]+par_c[ind2]<1.0:
    #         return 0.0
    #     else:
    #         return -np.inf
    # else:
    #     if all(np.greater(par_c,par_d)) and all(np.less(par_c,par_u)):
    #         return 0.0
    #     else:
    #         return -np.inf
    #
    par_c=theta
    if all(np.greater(par_c,par_d)) and all(np.less(par_c,par_u)):
        return 0.0
    else:
        return -np.inf

def lnlike(theta,t,flux_lc,err,model,par_n,par_d,par_u,ful_par_c,fit_inds,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0):
    par_c=theta
    fit=lc_func(t,model,ful_par_c,fit_inds,False,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0,*par_c)
    residuals_lc=flux_lc-fit
    return -0.5*(np.nansum((residuals_lc/err)**2))#-0.5*(np.nansum((residuals_cs/err)**2.))


def lnprob(theta,par_c,t,flux_lc,err,model,par_n,par_d,par_u,ful_par_c,fit_inds,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0):
    prior=lnprior(theta,par_n,par_d,par_u)
    # print(prior)
    if not np.isfinite(prior):
        return -np.inf
    post=prior+lnlike(theta,t,flux_lc,err,model,par_n,par_d,par_u,
               ful_par_c,fit_inds,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0)
    return post

#######################

def lc_plot(t,model,params,data,data_err,savename,color,figtext,par_n,ful_par_c,fit_inds,plot,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0):
    
    ### Function to plot the LC fits ###
    
    LC=data
    #t0=params[0]
    
    ful_par_c[fit_inds] = params
    
    lcf, basel, noise, FWC, BGC, XpC, YpC = lc_func(
        t,model,ful_par_c,fit_inds,plot,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0,params)
    
    
    fig,ax=plt.subplots(4,1,figsize=(8,20),sharex=True,gridspec_kw={'height_ratios': [3, 3, 3,1]})
    fig.subplots_adjust(wspace=0,hspace=0)
    plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.05)
    
    fig.suptitle(figtext,fontsize=25)

    ax[0].plot(24.*t,data,'.',ms=14,mfc='none',mec='black',alpha=0.8,mew=2)
    ax[0].plot(24.*t,lcf*basel*noise,ls='-',c=color,lw=8,alpha=0.6,zorder=0)

    #ax[1].plot(t,noise/np.nanmax(noise),ls='-',c='grey',lw=4,alpha=0.6,zorder=10)
    ax[1].plot(24.*t,FWC*(fwhm_)/np.nanmax(fwhm_),ls='--',color='darkgoldenrod',lw=2,alpha=0.8,label='fwhm')
    ax[1].plot(24.*t,XpC*(X_loc)/np.nanmax(X_loc),ls='--',color='dodgerblue',lw=2,alpha=0.8,label='X')
    ax[1].plot(24.*t,YpC*(Y_loc)/np.nanmax(Y_loc),ls='--',color='darkgreen',lw=2,alpha=0.8,label='Y')
    ax[1].plot(24.*t,BGC*(bg_ct)/np.nanmax(bg_ct),ls='--',color='brown',lw=2,alpha=0.8,label='bg')
    ax[1].legend(fontsize=14,loc='best')

    ax[2].axhline(y=1.0,ls='--',c='darkgrey',lw=2.0)
    ax[2].plot(24.*t,LC/(noise*basel),'.',ms=14,mfc='none',mec='black',alpha=0.8,mew=2)
    ax[2].plot(24.*t,lcf,ls='-',c=color,lw=8,alpha=0.6,zorder=0)

    ax[3].axhline(y=0.0,ls='--',c='darkgrey',lw=2.0)
    ax[3].plot(24.*t,LC-(lcf*basel*noise),'.',ms=12,mfc='none',mec='black',alpha=0.8,mew=2)



    ax[0].set_ylabel('Relative Flux', fontsize=20)
    ax[1].set_ylabel('Relative Contribution', fontsize=20)
    ax[2].set_ylabel('Relative Flux', fontsize=20)
    ax[3].set_ylabel('Residuals', fontsize=20)

    ax[3].set_xlabel('Time [hours]',fontsize=20)

 
    ax[0].set_title(" ".join(par_n),fontsize=15)
    
    
    ###
    residuals=lcf-LC/(noise*basel)
    chisq_red=(np.nansum(residuals**2./data_err**2.))/(len(residuals)-len(fit_inds))
    #print('              ...testttt:',np.nanmedian(err_ptn),np.nanmedian(residuals))
    BIC=(np.nansum(residuals**2./data_err**2.))+len(fit_inds)*np.log10(len(residuals))
    
    sdnr = np.std(residuals)*10**6.
    ###
    plt.figtext(0.65,0.3,'$\chi^2$ = ' + str(np.round(chisq_red,4)), fontsize=20)
    plt.figtext(0.65,0.28,'BIC = ' + str(np.round(BIC,4)), fontsize=20)
    plt.figtext(0.65,0.26,'sdnr = ' + str(np.round(sdnr,2)), fontsize=20)
    #plt.figtext(0.25,0.90,figtext,fontsize=35,weight='bold')

    plt.savefig(savename)
    #plt.draw()
    plt.close()
    
    return lcf*basel*noise,noise,basel


#######################

def lc_func(t,model,ful_par_c,fit_inds,plot,useFW,useBG,useXp,useYp,fwhm_,bg_ct,X_loc,Y_loc,fwhm0,bg0,x0,y0,*params):
#     Ms_s=Ms#np.random.normal(loc=Ms,scale=Ms_e)
#     Rs_s=Rs#np.random.normal(loc=Rs,scale=Rs_e)
#     smacm=(((per*24.*60.*60.)**2.*Grav*Ms_s)/(4*np.pi**2.))**(1./3.)
#     sma_f=smacm/Rs_s

    ful_par_c[fit_inds]=params

    lcparamsF=ful_par_c[:9]
    polyparams=ful_par_c[9:]
    
    #####
    t0, per, rp, sma, inc, ecc, w, q1,q2 = lcparamsF
    
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
        
    params=batman.TransitParams()
    params.t0=t0
    params.per=per
    params.rp=rp
    params.a=sma
    params.inc=inc
    params.ecc=ecc
    params.w=w
    params.u=[u1,u2]
    params.limb_dark='quadratic'
    
    fit=model.light_curve(params)
    #print(params.t0,params.per,params.rp,params.a,params.inc,params.u)
    
    nn = np.sum([useFW,useBG,useXp,useYp])
    FWC, BGC, XpC, YpC = 1.0, 1.0, 1.0, 1.0
    if nn>0:
        nn+=1 #account for constant
        pp,pt=polyparams[:(-1*nn)],polyparams[(-1*nn):]
        basel=(np.poly1d(pp))(t-t0)
        
        pt, CC = pt[:-1],pt[-1]
        if useYp==True:
            pt, YpC = pt[:-1],pt[-1]
        if useXp==True:
            pt, XpC = pt[:-1],pt[-1]
        if useBG==True:
            pt, BGC = pt[:-1],pt[-1]
        if useFW==True:
            pt, FWC = pt[:-1],pt[-1]
        
        noise=np.nansum([FWC*(fwhm_-fwhm0),BGC*(bg_ct-bg0),XpC*(X_loc-x0),YpC*(Y_loc-y0), CC])
    else:
        pp=polyparams
        basel=(np.poly1d(pp))(t-t0)
        noise=np.ones(len(t))

    if plot==False:
        # # if c1>c2 and c1+c2<1.0:
        # if c1+c2<1.0:
        #     return fit*basel*noise
        # else:
        #     return np.zeros(len(t))
        return fit*basel*noise
    elif plot==True:
        return fit, basel, noise, FWC, BGC,XpC, YpC
    
    
##############
# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Automated windowing procedure following Sokal (1989)
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

