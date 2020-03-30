##################################################
# Performs Lightcurve Fits for the MOPSS program #
##################################################
import numpy as np

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

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import sys

import os
from os import path

from datetime import datetime

from outlier_removal import outlierr_w

########
import batman
import emcee
import corner

########

startT1=datetime.now()

######## Below: Reads in input file, imports data ######

from LCFITS_in import *
from LCFITS_funcs import *

print(' ')
print('----------------------------')
print('      PLANET: ', p_name)
print('    OBS_DATE: ', obs_date)
print('----------------------------')
print('Input Period (days): ', per)
print('Input Rp (rp/rs):    ', rp)
print('Input SMA (sma/rs):  ', sma)
print('----------------------------')

t=np.array(np.load(fullpath+'/Obs_times.npz')['times'])
n_exp=len(t)

print('Number of Exposures= ',n_exp)


limb_dark='quadratic'
wave=[5230, 5700, 6170, 6640, 7110, 7580, 8050, 8520, 8990, 9460]#,16300,21900]                                                                              
# q0_func=interp1d(wave,q0)
# q1_func=interp1d(wave,q1)
u1_start = np.array(q0)
u2_start = np.array(q1)
q1 = (u1_start+u2_start)**2
q2 = u1_start / (2*(u1_start+u2_start))
q1_func=interp1d(wave,q1)
q2_func=interp1d(wave,q2)

norm=matplotlib.colors.Normalize(vmin=4000,vmax=9000)
colors=matplotlib.cm.jet
scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
scal_m.set_array([])


if c==0:
    bins_center=np.array([6000.])
    LC=np.load(fullpath+'LCwhite_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['data']
    err_ptn=np.load(fullpath+'LCwhite_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['err_p']#np.ones(len(LC))*10**-4.#
    
    model_inputs=np.load(fullpath+'NoiseModel_Inputs_'+str(int(width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')
    
    o=0
    if useXp==True:
        X_loc=((model_inputs['white_x'])[o,:]-(model_inputs['white_x'])[o,0])
    else:
        X_loc=np.ones_like(t)*0.0
    if useYp==True:
        Y_loc=((model_inputs['yshift'])[o,:])
    else:
        Y_loc=np.ones_like(t)*0.0
    if useBG==True:
        bg_ct=((model_inputs['white_bg'])[o,:])
        if np.nanmax(bg_ct) != 0:
            bg_ct /= np.nanmax(bg_ct)
    else:
        bg_ct=np.ones_like(t)*0.0
    
else:
    bins_center=np.load(fullpath+'/LC_bins_'+str(int(width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['bin_ctr']
    LC_bins=np.load(fullpath+'LC_bins_'+str(int(width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['data']
    err_ptn_bins=np.load(fullpath+'LC_bins_'+str(int(width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['err_p']#np.ones(len(LC_bins[:,0]))*10**-4.#
    
    model_inputs=np.load(fullpath+'NoiseModel_Inputs_'+str(int(width))+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')
    
    o=0
    if useXp==True: 
        X_loc_bins=((model_inputs['binned_x'])[o,:,:]-(model_inputs['binned_x'])[o,0,:])
    else:
        X_loc_bins=np.ones_like(LC_bins)*0.0
    if useYp==True:
        Y_loc=((model_inputs['yshift'])[o,:])
    else:
        Y_loc=np.ones_like(t)*0.0
    if useBG==True:
        bg_ct_bins=((model_inputs['binned_bg'])[o,:,:])
    else:
        bg_ct_bins=np.ones_like(LC_bins)*0.0

n_bins=len(bins_center)
if useFW==True:
    fwhm_ar=np.load(fullpath+'2DFit_Obj0.npz')['gaus'][:,:,2]  #(np.load(fullpath+'FlattenedSpectra.npz')['fwhm_ar'])[o,:]
    fwhm_ = 2.*np.sqrt(2.*np.log(2.))*np.nanmedian(fwhm_ar,axis =1)
    
    # this could be updated later -- if CON == 2 then the seeing changes with wavelength where taken into account with the LCs
else:
    fwhm_=np.ones_like(t)*0.0

print('Number of Exposures= ',n_exp)
if c==1:
    print('Number of Bins =     ',n_bins)
    print('Bin Width =          ', width)

########

ful_par_n = np.array(['t0', 'per', 'rp', 'sma', 'inc', 'ecc', 'w', 'q1', 'q2'])
ful_par_c = np.ones_like(ful_par_n,dtype=float)
#fill in full array
ful_par_c = t0, per, rp, sma, inc, ecc, w, 0.5, 0.5

nlcp = len(par_n)
flcp = len(ful_par_n)


npars = nlcp+(BL_order+1)  #number of LC and BL params

#append initial guesses and param names for BL function
par_c = np.append(in_par_c,np.ones(BL_order+1))  
par_d = np.append(in_par_d,np.ones(BL_order+1)*-100.)
par_u = np.append(in_par_u,np.ones(BL_order+1)*100.)
ful_par_c = np.append(ful_par_c,np.ones(BL_order+1))  

for i in range(0,BL_order+1):
    if i==BL_order:       #(last param is the constant)
        par_n = np.append(par_n,'bc')
        ful_par_n = np.append(ful_par_n,'bc')
    else:
        par_n = np.append(par_n,'b'+str(int(i+1)))
        ful_par_n = np.append(ful_par_n,'b'+str(int(i+1)))

if useFW==True:
    npars+=1
    par_c=np.append(par_c,[1.0*10**-5])
    par_d=np.append(par_d,[-1.0*10**-2])
    par_u=np.append(par_u,[1.0*10**-2])
    par_n=np.append(par_n,'FWC')
    ful_par_n=np.append(ful_par_n,'FWC')
    ful_par_c=np.append(ful_par_c,[1.0*10**-5])
if useBG==True:
    npars+=1
    par_c=np.append(par_c,[1.0*10**-5])
    par_d=np.append(par_d,[-1.0*10**-2])
    par_u=np.append(par_u,[1.0*10**-2])
    par_n=np.append(par_n,'BGC')
    ful_par_n=np.append(ful_par_n,'BGC')
    ful_par_c=np.append(ful_par_c,[1.0*10**-5])
if useXp==True:
    npars+=1
    par_c=np.append(par_c,[1.0*10**-5])
    par_d=np.append(par_d,[-1.0*10**-2])
    par_u=np.append(par_u,[1.0*10**-2])
    par_n=np.append(par_n,'XpC')
    ful_par_n=np.append(ful_par_n,'XpC')
    ful_par_c=np.append(ful_par_c,[1.0*10**-5])
if useYp==True:
    npars+=1
    par_c=np.append(par_c,[1.0*10**-5])
    par_d=np.append(par_d,[-1.0*10**-2])
    par_u=np.append(par_u,[1.0*10**-2])
    par_n=np.append(par_n,'YpC')
    ful_par_n=np.append(ful_par_n,'YpC')
    ful_par_c=np.append(ful_par_c,[1.0*10**-5])

if any([useFW,useXp,useYp]):  #if any noise models used, add a constant offset
    # npars+=1
    # par_c=np.append(par_c,1.0)
    # par_d=np.append(par_d,0.0)
    # par_u=np.append(par_u,1.0*10**2)
    # par_n=np.append(par_n,'CC')
    ful_par_n=np.append(ful_par_n,'CC')
    ful_par_c=np.append(ful_par_c,1.0)


ndim=npars

normA=matplotlib.colors.Normalize(vmin=0,vmax=len(par_c))
colorsA=matplotlib.cm.viridis
scal_mA=matplotlib.cm.ScalarMappable(cmap=colorsA,norm=normA)
scal_mA.set_array([])


print('----------------------------')
print('Baseline Order =     ',BL_order)
print('Use FWHM?            ',useFW)
print('Use spatial centroid?',useXp)
print('Use spectra centroid?',useYp)
print(' ')
print('  *FIT PARAM NAMES*  ')
print('   ',par_n) 
print('----------------------------')
print(' ')

fit_inds = np.empty(len(par_n),dtype=int)
for ni,n in enumerate(par_n):
    fit_inds[ni]=int(np.where(ful_par_n==n)[0])

print('... checking dimensions')
if par_d.shape[0] == npars and par_c.shape[0] == npars and par_u.shape[0] == npars:
    print('   PASS!')
    print('      ', npars, par_d.shape[0], par_c.shape[0], par_u.shape[0])
else:
    print('    ERROR: shape mismatch!')
    print(npars, par_d.shape,par_c.shape,par_u.shape)

print(' ')
print('----------------------------')

fullpath=fullpath+'LCFITS/'+mydir+'/'


fitparams_allbins=np.empty([len(bins_center),len(ful_par_c)])
N_acor = np.exp(np.linspace(np.log(100), np.log(nsteps), 100)).astype(int)

for bi,b in enumerate(bins_center):
    fignum=1
    if c==0:
        print('--- RUNNING BIN: White LC')
    else:
        print('--- RUNNING BIN: ', b)
        
    if len(bins_center)>1:
        LC=LC_bins[:,bi]
        err_ptn=err_ptn_bins[:,bi]
        X_loc=X_loc_bins[:,bi]
        bg_ct=bg_ct_bins[:,bi]
        if useBG==True:
            bg_ct/=np.nanmax(bg_ct)
            
            
    if outlier == True:  #remove and mask outliers
        print('   --- masking outliers')
        mask_indexes = outlierr_w(LC,ks,sig)
        print('             ', np.where(~mask_indexes)[0])
        t_fitar              = t[mask_indexes]
        LC_fitar             = LC[mask_indexes]
        err_ptn_fitar        = err_ptn[mask_indexes]
        X_loc_fitar          = X_loc[mask_indexes]
        Y_loc_fitar          = Y_loc[mask_indexes]
        bg_ct_fitar          = bg_ct[mask_indexes]
        fwhm_fitar           = fwhm_[mask_indexes]
        if runtwce == True:
            print('   --- masking outliers 2')
            mask_indexes_2 = outlierr_w(LC_fitar,ks,sig)
            print('             ', np.where(~mask_indexes_2)[0])
            t_fitar              = t_fitar[mask_indexes_2]
            LC_fitar             = LC_fitar[mask_indexes_2]
            err_ptn_fitar        = err_ptn_fitar[mask_indexes_2]
            X_loc_fitar          = X_loc_fitar[mask_indexes_2]
            Y_loc_fitar          = Y_loc_fitar[mask_indexes_2]
            bg_ct_fitar          = bg_ct_fitar[mask_indexes_2]
            fwhm_fitar           = fwhm_fitar[mask_indexes_2]
    else:
        t_fitar              = np.copy(t)
        LC_fitar             = np.copy(LC)
        err_ptn_fitar        = np.copy(err_ptn)
        X_loc_fitar          = np.copy(X_loc)
        Y_loc_fitar          = np.copy(Y_loc)
        bg_ct_fitar          = np.copy(bg_ct)
        fwhm_fitar           = np.copy(fwhm_)
        

    q1=q1_func(b)
    q2=q2_func(b)
    
    print('Wavelength - q1 - q2:')
    print(b,q1,q2)
    
    if 'q1' in par_n:
        par_c[np.where(par_n=='q1')[0]]=q1
        par_d[np.where(par_n=='q1')[0]]=0.0
        par_u[np.where(par_n=='q1')[0]]=1.0
        print('     ..... q1 updated')
    else:
        ful_par_c[np.where(ful_par_n=='q1')[0]]=q1
        print('     ..... q1 updated')
    if 'q2' in par_n:
        par_c[np.where(par_n=='q2')[0]]=q2
        par_d[np.where(par_n=='q2')[0]]=0.0
        par_u[np.where(par_n=='q2')[0]]=1.0
        print('     ..... q2 updated')
    else:
        ful_par_c[np.where(ful_par_n=='q2')[0]]=q2
        print('     ..... q2 updated')
    
    ful_par_c[fit_inds]=par_c
    

      
    ##### initialize transit function #####
    
    t0, per, rp, sma, inc, ecc, w, q1,q2 = ful_par_c[:flcp]
    
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
    
    print('Wavelength - U1 - U2')
    print(b, u1, u2)
        
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
    
    model=batman.TransitModel(params,t_fitar)
    LC0=np.copy(model.light_curve(params))
    
    ############  LEAST SQUARED FIT ##########
    
    initial,inunc=curve_fit(
        lambda t,*params: lc_func(t_fitar,model,ful_par_c,fit_inds,False,useFW,useBG,useXp,useYp,fwhm_fitar,bg_ct_fitar,X_loc_fitar,Y_loc_fitar,fwhm0,bg0,x0,y0,params),
        t_fitar,LC_fitar,
        p0=par_c,bounds=(par_d,par_u),maxfev=5000)
    
    iner=np.sqrt(np.diag(inunc))
    #print(initial)


    #t0=initial[0]
    if c==0:
        figtext='WHITE LC - LS'
        figname= fullpath+'LCfit_plots/LC_plots_leastsquare_whitelc.png'
    else:
        figtext=str(int(b))+' - LS'
        figname= fullpath+'LCfit_plots/LC_plots_leastsquare_'+str(int(width))+'_bin'+str(int(b))+'.png'
    ls_LCFIT,ls_NOISEFIT,ls_BLFIT=lc_plot(t_fitar,model,initial,LC_fitar,err_ptn_fitar,
                                          figname,scal_m.to_rgba(b),figtext,par_n,
                                          ful_par_c,fit_inds,True,useFW,useBG,useXp,useYp,fwhm_fitar,bg_ct_fitar,X_loc_fitar,Y_loc_fitar,fwhm0,bg0,x0,y0)
    
    if c==0:
        filename=fullpath+'LCfit_files/leastsquared_run_whitelc.npz'
    else:
        filename=fullpath+'LCfit_files/leastsquared_run_'+str(int(width))+'_bin'+str(int(b))+'.npz'
        
    ful_par_c[fit_inds]=initial
    fitparams_allbins[bi,:]=ful_par_c
    np.savez(filename,
             best_params=initial,par_n=par_n,bpe=iner, lc=ls_LCFIT, noise=ls_NOISEFIT, bl=ls_BLFIT,ful_pars=ful_par_c)
    
    
    
    ########## MCMC SET UP ##########
    #generate initial positions within X sigma of best fits from least-squared minimilazation
    sig=3.
    pos0=np.empty([nwalkers,ndim])
    for j in range(ndim):
        d=np.nanmax([initial[j]-sig*iner[j],par_d[j]])
        u=np.nanmin([initial[j]+sig*iner[j],par_u[j]])
        #print(par_n[j],par_c[j],par_d[j],par_u[j],initial[j],d,u)
        arr = np.linspace(d,u,10**6.)
        for i in range(nwalkers):
            pos0[i,j]=np.random.choice(arr)
            
            
    #save initial chain
    if c==0:
        filename=fullpath+'LCfit_files/InitialPositions_whitelc.npz'
    else:
        filename=fullpath+'LCfit_files/InitialPositions_'+str(int(width))+'_bin'+str(int(b))+'.npz'
    np.savez(filename,pos0=pos0)

    #initialize emcee chain
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,a=5.0,args=(par_c,t_fitar,LC_fitar,err_ptn_fitar,model,par_n,par_d,par_u,ful_par_c,fit_inds,
                                                                   useFW,useBG,useXp,useYp,fwhm_fitar,bg_ct_fitar,X_loc_fitar,Y_loc_fitar,fwhm0,bg0,x0,y0))
    
    sampler.reset()

    print('    ...running burn-in  ')
    time1=datetime.now()
    p0,hold1,hold2=sampler.run_mcmc(pos0,nburnin)
    time2=datetime.now()
   

    print('           Time to Run: ', time2-time1)
   
    del pos0
    
    #rescale errors  #############
    
    samples = sampler.flatchain
    best_parC_burn = samples[np.argmax(sampler.flatlnprobability)]
    if c==0:
        figtext='WHITE LC - BI'
        figname= fullpath+'LCfit_plots/LC_plots_mcmc_burnin_whitelc.png'
    else:
        figtext=str(int(b))+' - BI'
        figname= fullpath+'LCfit_plots/LC_plots_mcmc_burnin_'+str(int(width))+'_bin'+str(int(b))+'.png'
    b_LCFIT,b_NOISEFIT,b_BLFIT=lc_plot(t_fitar,model,initial,LC_fitar,err_ptn_fitar,
                                          figname,scal_m.to_rgba(b),figtext,par_n,
                                          ful_par_c,fit_inds,True,useFW,useBG,useXp,useYp,fwhm_fitar,bg_ct_fitar,X_loc_fitar,Y_loc_fitar,fwhm0,bg0,x0,y0)
    
    residuals=LC_fitar-b_LCFIT
    chisq_red=(np.nansum(residuals**2./err_ptn_fitar**2.))/(len(residuals)-len(fit_inds))
    #print('              ...testttt:',np.nanmedian(err_ptn),np.nanmedian(residuals))
    print('              ...current reduced chi2:', chisq_red)
    err_ptn_fitar*=np.sqrt(chisq_red)
    #print('              ...testttt:',np.nanmedian(err_ptn),np.nanmedian(residuals))

    ###############################
    
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,a=5.0,args=(par_c,t_fitar,LC_fitar,err_ptn_fitar,model,par_n,par_d,par_u,ful_par_c,fit_inds,
                                                                   useFW,useBG,useXp,useYp,fwhm_fitar,bg_ct_fitar,X_loc_fitar,Y_loc_fitar,fwhm0,bg0,x0,y0))
    sampler.reset()
    print('    ...running chain  ' )       

    for i, result in enumerate(sampler.sample(p0,iterations=nsteps)):
        if (i+1)%(nsteps/10) ==0:
            print("            ",(float(i+1) / nsteps),'          ', datetime.now())

    #######
    #['t0' 'per' 'rp' 'inc' 'b1' 'bc' 'FWC' 'XpC' 'XpC' 'CC']
    #print(par_n)
    print(np.mean(sampler.acceptance_fraction))

    samples = sampler.flatchain

    ###### separate best params
    best_parC = samples[np.argmax(sampler.flatlnprobability)]
    #best_parC=np.ones(len(par_c))*1.0
    best_parD=np.ones(len(par_c))*1.0
    best_parU=np.ones(len(par_c))*1.0
    best_parH=np.ones(len(par_c))*1.0
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(par_n[i],mcmc[1],q[0],q[1])
        best_parH[i],best_parD[i],best_parU[i]=mcmc[1],q[0],q[1]


    #t0=best_parC[0]
    ful_par_c[fit_inds]=best_parC
    fitparams_allbins[bi,:]=ful_par_c
     
    if c==0:
        figtext='WHITE LC - MCMC'
        figname= fullpath+'LCfit_plots/LC_plots_mcmc_prod_whitelc.png'
    else:
        figtext=str(int(b))+' - MCMC'
        figname= fullpath+'LCfit_plots/LC_plots_mcmc_prod_'+str(int(width))+'_bin'+str(int(b))+'.png'
    LCFIT,NOISEFIT,BLFIT=lc_plot(t_fitar,model,best_parC,LC_fitar,err_ptn_fitar,
                                          figname,scal_m.to_rgba(b),figtext,par_n,
                                          ful_par_c,fit_inds,True,useFW,useBG,useXp,useYp,fwhm_fitar,bg_ct_fitar,X_loc_fitar,Y_loc_fitar,fwhm0,bg0,x0,y0)
    
    
    if c==0:
        filename=fullpath+'LCfit_files/emcee_run_whitelc.npz'
        np.savez(fullpath0+'/emcee_run_whitelc_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz',
             chain=samples,best_params=best_parC, eU = best_parU, eD = best_parD ,par_n=par_n,
             lc=LCFIT,noise=NOISEFIT, bl= BLFIT)
    else:
        filename=fullpath+'LCfit_files/emcee_run_'+str(int(width))+'_bin'+str(int(b))+'.npz'
    
    #print(best_parD)
    
    
    #print(fitparams_allbins[bi,:])
             
    

    ## corner plot###
    if c==0:
        figtext='WHITE LC - MCMC'
        figname= fullpath+'LCfit_plots/EMCEE_corner_whitelc.png'
    else:
        figtext=str(int(b))+' - MCMC'
        figname= fullpath+'LCfit_plots/EMCEE_corner_'+str(int(width))+'_bin'+str(int(b))+'.png'

    plt.figure(fignum+1)
    corner.corner(samples,labels=par_n,truths=initial,label_kwargs=dict(fontsize=25,weight='bold'))
    plt.figtext(0.75,0.75,figtext,fontsize=45,ha='center',va='center')
    plt.savefig(figname)

    #plt.show()
    plt.close(fignum+1)
    
    residuals=LC_fitar-LCFIT
    chisq_red_new=(np.nansum(residuals**2./err_ptn_fitar**2.))/(len(residuals)-len(fit_inds))
    #print('              ...testttt:',np.nanmedian(err_ptn),np.nanmedian(residuals))
    print('              ...current reduced chi2:', chisq_red_new)
    BIC=(np.nansum(residuals**2./err_ptn_fitar**2.))+len(fit_inds)*np.log10(len(residuals))
    print('                       ...current BIC:', BIC)
    sdnr = np.std(residuals)*10**6.
    print('                       ...current SDNR:', sdnr)
    
    
    np.savez(filename,
             chain=samples,best_params=best_parC, eU = best_parU, eD = best_parD ,bpH=best_parH,par_n=par_n,
             lc=LCFIT,noise=NOISEFIT, bl= BLFIT,ful_pars=ful_par_c,chisq_red_new=chisq_red_new,chisq_red_org=chisq_red,BIC=BIC)

    print('DONE!')
    
    del samples
    
    print('Calculating AutoCor Times..')
    R_acor = np.empty([len(par_c),len(N_acor)])*np.nan
    for k in range(0,len(par_c)):
        for i, n in enumerate(N_acor):
            R_acor[k,i] = autocorr_gw2010(sampler.chain[:, :n,k])
    
  
    del sampler
    
    plt.figure(fignum+1,figsize=(8,6))
    plt.subplots_adjust(right=0.7)
    plt.clf()
    for k in range(0,len(par_c)):
        plt.plot(N_acor,R_acor[k,:],'-',lw=3.0,color=scal_mA.to_rgba(k),label=par_n[k])
    plt.plot(N_acor, N_acor / 100.0, "--k", label=r"$\tau = N/100$")
    plt.plot(N_acor, N_acor / 50.0, "--",color='grey', label=r"$\tau = N/50$")
    plt.plot(N_acor, N_acor / 10.0, "-",color='black', linewidth=2.5,label=r"$\tau = N/10$")

    plt.xlabel("Number of Steps",fontsize=25)
    plt.ylabel("Autucorrelation Time",fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=14,bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    if c==0:
        figname= fullpath+'LCfit_plots/AutoCor_whitelc.png'
    else:
        figname= fullpath+'LCfit_plots/AutoCor_'+str(int(width))+'_bin'+str(int(b))+'.png'

    plt.savefig(figname)
    plt.close(fignum+1)

    
    
if c>0:
    filename=fullpath+'LCfit_files/emcee_run_ALL_bin'+str(int(b))+'.npz'
    np.savez(filename,allparams=fitparams_allbins)

    
if len(bins_center)>1:
    rprs_ls=np.ones_like(bins_center)*1.0
    rprs_mc=np.ones_like(bins_center)*1.0
    rprs_dn=np.ones_like(bins_center)*1.0
    rprs_up=np.ones_like(bins_center)*1.0
    q1_mc=np.ones_like(bins_center)*1.0
    q2_mc=np.ones_like(bins_center)*1.0
    rpind_j=np.where(par_n=='rp')[0]
    #print(rpind_j)
    rpind=np.where(ful_par_n=='rp')[0]
    q1ind=np.where(ful_par_n=='q1')[0]
    q2ind=np.where(ful_par_n=='q2')[0]
    for bi,b in enumerate(bins_center):
        rprs_mc[bi]=np.load(fullpath
                            +'LCfit_files/emcee_run_'+str(int(width))+'_bin'+str(int(b))+'.npz')['ful_pars'][rpind]
        rprs_dn[bi]=np.load(fullpath
                            +'LCfit_files/emcee_run_'+str(int(width))+'_bin'+str(int(b))+'.npz')['eD'][rpind_j]
        rprs_up[bi]=np.load(fullpath
                            +'LCfit_files/emcee_run_'+str(int(width))+'_bin'+str(int(b))+'.npz')['eU'][rpind_j]
        q1_mc[bi]=np.load(fullpath
                            +'LCfit_files/emcee_run_'+str(int(width))+'_bin'+str(int(b))+'.npz')['ful_pars'][q1ind]
        q2_mc[bi]=np.load(fullpath
                            +'LCfit_files/emcee_run_'+str(int(width))+'_bin'+str(int(b))+'.npz')['ful_pars'][q2ind]
        rprs_ls[bi]=np.load(fullpath
                            +'LCfit_files/leastsquared_run_'+str(int(width))+'_bin'+str(int(b))+'.npz')['ful_pars'][rpind]
    
    #print(rprs_dn,rprs_up)
    plt.figure(101,figsize=(10,6))

    ms=35
    for bi,b in enumerate(bins_center):
        plt.plot(bins_center[bi],rprs_mc[bi],'.',mfc=scal_m.to_rgba(b),mec='none',ms=ms,alpha=0.4)
        plt.plot(bins_center[bi],rprs_mc[bi],'.',mfc='none',mec=scal_m.to_rgba(b),mew=4.0,ms=ms)
        plt.errorbar(bins_center[bi],rprs_mc[bi],yerr=np.array([[rprs_dn[bi],rprs_up[bi]]]).T,color='black')
    
    plt.xlabel('Wavelength, A',fontsize=20)
    plt.ylabel('Rp/Rs',fontsize=20)
    plt.xscale('log')
    plt.savefig(fullpath+'LCfit_plots/Amc_quicklook_tspec.png')
    plt.close()
    
    plt.figure(102,figsize=(10,6))

    ms=35
    for bi,b in enumerate(bins_center):
        plt.plot(bins_center[bi],rprs_ls[bi],'.',mfc=scal_m.to_rgba(b),mec='none',ms=ms,alpha=0.4)
        plt.plot(bins_center[bi],rprs_ls[bi],'.',mfc='none',mec=scal_m.to_rgba(b),mew=4.0,ms=ms)
    
    plt.xlabel('Wavelength, A',fontsize=20)
    plt.ylabel('Rp/Rs',fontsize=20)
    plt.xscale('log')
    plt.savefig(fullpath+'LCfit_plots/Als_quicklook_tspec.png')
    plt.close()
    
    #####
    plt.figure(103,figsize=(10,6))
    ms=35
    u1_mc = 2*np.sqrt(q1_mc)*q2_mc
    u2_mc = np.sqrt(q1_mc)*(1-2*q2_mc)
    for bi,b in enumerate(bins_center):
        plt.plot(bins_center[bi],u1_mc[bi],'.',mfc=scal_m.to_rgba(b),mec='none',ms=ms,alpha=0.4)
        plt.plot(bins_center[bi],u1_mc[bi],'.',mfc='none',mec=scal_m.to_rgba(b),mew=4.0,ms=ms)
        
        
        plt.plot(bins_center[bi],u2_mc[bi],'.',mfc=scal_m.to_rgba(b),mec='none',ms=ms,alpha=0.4)
        plt.plot(bins_center[bi],u2_mc[bi],'.',mfc='none',mec=scal_m.to_rgba(b),mew=4.0,ms=ms)
    
    plt.plot(bins_center,u1_mc,'-',color='grey',lw=0.5)
    plt.plot(bins_center,u2_mc,'-',color='grey',lw=0.5)
    plt.xlabel('Wavelength, A',fontsize=20)
    plt.ylabel('u1/u2',fontsize=20)
    plt.xscale('log')
    plt.savefig(fullpath+'LCfit_plots/Amc_quicklook_C1C2.png')
    plt.close()
    
    
    
    
        
      
        
        
