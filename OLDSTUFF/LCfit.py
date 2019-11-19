#####################################################                                                                                      
#####################################################                                                                                           
##        Fitting of lightcurves using BATMAN      ##                                                                                           
##  ---------------------------------------------  ##                                                                                           
##  reads in .npz files from Lightcurve_Script.py  ##                                                                                           
##     See README.txt in Magellan/pipeline         ##                                                                                           
#####################################################                                                                                           
#####################################################   

import numpy as np
np.seterr(all='ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import batman

import emcee
import corner

#import astropy
#from astropy.time import Time
#import pyfits                                                                                                                                  

import scipy
from scipy.interpolate import interp1d

from datetime import datetime
startTime1=datetime.now()

import sys
print (sys.argv)
 


#width=200


####################################################                                                                                            
#              LIGHT CURVE FUNCTION                #                                                                                            
####################################################                                                                                            
def lc_func(t,t0,per,rp,a,inc,ecc,w,u,limb_dark):
    params=batman.TransitParams()
    params.t0=t0
    params.per=per
    params.rp=rp
    params.a=a
    params.inc=inc
    params.ecc=ecc
    params.w=w
    params.u=u
    params.limb_dark=limb_dark

    model=batman.TransitModel(params,t)
    return params, model

####################################################                                                                                            
#     Prior, Likelihood, Posterior Functions       #                                                                                            
####################################################                                                                                            
def loggaus(mean,sig,param):
    return -np.log(np.sqrt(2*np.pi*sig**2.))-((param-mean)**2.)/(2*sig**2.)

def lnprior(SAVEPATH,theta,initial,color):
    sys.path.insert(0,SAVEPATH) 
    from SystemCons import *
    
    if color==0:
        t0,per,rp,inc,c1,c2=initial
        t0_f,per_f,rp_f,inc_f,c1_f,c2_f=theta
        if t0_f>t0dn and t0_f<t0up and rp_f>rpdn and rp_f<rpup and np.abs(c2_f-c2)<0.1 and np.abs((c2_f-c2)-(c1_f-c1))<0.05:
            return loggaus(per,per_e,per_f)+loggaus(inc,inc_e,inc_f)
        else:
            return -np.inf
    if color==1:
        rp,c1,c2=initial
        rp_f,c1_f,c2_f=theta
        if rp_f<rpup and rp_f>rpdn and np.abs(c2_f-c2)<0.1 and np.abs((c2_f-c2)-(c1_f-c1))<0.05:
            return 0.0
        else:
            return -np.inf

def lnlike(SAVEPATH,theta,params,model,t,flux,err,color):
    sys.path.insert(0,SAVEPATH) 
    from SystemCons import *
    
    if color==0:
        params.t0,params.per,params.rp,params.inc,c1_f,c2_f=theta
        params.u=[c1_f,c2_f]
        Ms_s=np.random.normal(loc=Ms,scale=Ms_e)
        Rs_s=np.random.normal(loc=Rs,scale=Rs_e)
        smacm=(((params.per*24.*60.*60.)**2.*Grav*Ms_s)/(4*np.pi**2.))**(1./3.)
        params.a=smacm/Rs_s
        fit=model.light_curve(params)
        residuals=flux-fit
        return -0.5*(np.nansum((residuals/err)**2))
    if color==1:
        params.rp,c1_f,c2_f=theta
        params.u=[c1_f,c2_f]
        fit=model.light_curve(params)
        residuals=flux-fit
        return -0.5*(np.nansum((residuals/err)**2))


def lnprob(theta,initial,params,model,t,flux,err,color,SAVEPATH):
    prior=lnprior(SAVEPATH,theta,initial,color)
    if not np.isfinite(prior):
        return -np.inf
    post=prior+lnlike(SAVEPATH,theta,params,model,t,flux,err,color)
    return post

####################################################                                                                                            
#     RUN MCMC CODE      #                                                                                            
####################################################   

def runmcmc(SAVEPATH,width,ndim,nwalkers,burnin,nsteps,pos0,initial,params,model,time,data,error,color):
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,a=2.0, args=(initial,params,model,time,data,error,color,SAVEPATH))
    
    Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
    Output.write('{0} \n'.format('     -->> Running Burn-in...'))
                                                                                                                        
    time1=datetime.now()
    p0,test1,test2=sampler.run_mcmc(pos0,burnin)
    time2=datetime.now()
    Output.write('{0} {1} \n'.format('           Time to Run: ', time2-time1))
    sampler.reset()
                                                                                                                          

    Output.write('{0} \n'.format('     -->> Running Chain...'))
    Output.close()
    for i, result in enumerate(sampler.sample(p0,iterations=nsteps)):
        if (i+1)%(nsteps/10) ==0:
            Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
            Output.write('{0} {1} {2} \n'.format(("            {0:5.1%}".format(float(i+1) / nsteps)),'          ', datetime.now()))
            Output.close()

    Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
    Output.write('{0} {1} \n'.format('     -->> Mean Acceptance Fraction: ', np.mean(sampler.acceptance_fraction)))
    Output.close()
    
    samples=sampler.chain[:,:,:].reshape((-1,pos0.shape[1]))
    
    return samples

   

#################################
def lcfit(SAVEPATH,width,corr,avg,nwalkers,burnin,nsteps,color):
    
    sys.path.insert(0,SAVEPATH) 
    from SystemCons import *

    fulltime=np.load(SAVEPATH+'Obs_times.npz')['times']

    if corr==True:
        if avg==True:
            t=np.load(SAVEPATH+'LCwhite_br_Corr.npz')['avt']
            n_exp=len(t)
            
            lc_data_white=np.load(SAVEPATH+'LCwhite_br_Corr.npz')['avf']
            lc_data_binns=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'_Corr.npz')['avf']
            yerr_white=np.ones_like(lc_data_white)*np.nanmedian(np.load(SAVEPATH+'LCwhite_br_Corr.npz')['err_t'])
            yerr_binns=np.ones_like(lc_data_binns)*np.nanmedian(np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'_Corr.npz')['err_t'],axis=0)
            
        else:
            t=np.load(SAVEPATH+'Obs_times.npz')['times']
            n_exp=len(t)
            
            lc_data_white=np.load(SAVEPATH+'LCwhite_br_Corr.npz')['data']
            lc_data_binns=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'_Corr.npz')['data']
            yerr_white=np.load(SAVEPATH+'LCwhite_br_Corr.npz')['err_t']
            yerr_binns=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'_Corr.npz')['err_t']
    else:
        if avg==True:
            t=np.load(SAVEPATH+'LCwhite_br.npz')['avt']
            n_exp=len(t)
            
            lc_data_white=np.load(SAVEPATH+'LCwhite_br.npz')['avf']
            lc_data_binns=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz')['avf']
            yerr_white=np.ones_like(lc_data_white)*np.nanmedian(np.load(SAVEPATH+'LCwhite_br.npz')['err_t'])
            yerr_binns=np.ones_like(lc_data_binns)*np.nanmedian(np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz')['err_t'],axis=0)
            
        else:   
            t=np.load(SAVEPATH+'Obs_times.npz')['times']
            n_exp=len(t)
    
            lc_data_white=np.load(SAVEPATH+'LCwhite_br.npz')['data']
            lc_data_binns=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz')['data']
            yerr_white=np.load(SAVEPATH+'LCwhite_br.npz')['err_t']
            yerr_binns=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz')['err_t']
    
    bin_ctr=np.load(SAVEPATH+'LC_bins_br_'+str(int(width))+'.npz')['bin_ctr']
                
    ####################################################                                                                        
    #              LIMB DARKENING FUNCION              #                                                                        
    ####################################################                                                                          
    # U B V R I J H K                                                                                                            
    limb_dark='quadratic'
    wave=[3640,4450,5510,6580,8060,12200]#,16300,21900]                                                                          
    q0_func=interp1d(wave,q0)
    q1_func=interp1d(wave,q1)
    ####################################################                                                                                            
    norm=matplotlib.colors.Normalize(vmin=np.min(bin_ctr),vmax=np.max(bin_ctr))                                                                                                                 
    colors=matplotlib.cm.Spectral_r
    scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
    scal_m.set_array([])

    #####################
    nwalkers=nwalkers
    burnin=burnin
    nsteps=nsteps

    Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','w')

    if color==0:
        Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
        Output.write('{0}\n'.format('>>>>>>>>>> WHITE LIGHT CURVE <<<<<<<<<<'))
        print('>>>>>>>>>> WHITE LIGHT CURVE <<<<<<<<<<')
    
        c1=q0_func(6000)
        c2=q1_func(6000)
        u=[c1,c2]
                                                                                                                          
        params,m=lc_func(t,t0,per,rp,sma,inc,ecc,w,u,limb_dark)
    
        initial=np.array([t0,per,rp,inc,c1,c2])
        ndim=len(initial)
    
        Output.write('{0} {1} \n'.format('     -->> Initial Guess: ', initial))
        Output.close()

        t0_arr=np.linspace(t0dn,t0up,10**6.)
        per_arr=np.linspace(per-5*per_e,per+5*per_e,10**6.)
        rp_arr=np.linspace(rpdn,rpup,10**6.)                                                                                              
        inc_arr=np.linspace(inc-5*inc_e,inc+5*inc_e,10**6.)
        c1_arr=np.linspace(c1-0.2,c1+0.2,10**6.)
        c2_arr=np.linspace(c2-0.2,c2+0.2,10**6.)  

        pos0=np.empty([nwalkers,ndim])
        for i in range(nwalkers):
            pos0[i,:]=np.array([np.random.choice(t0_arr),np.random.choice(per_arr),np.random.choice(rp_arr),np.random.choice(inc_arr),np.random.choice(c1_arr),np.random.choice(c2_arr)])
        
        runwhite=runmcmc(SAVEPATH,width,ndim,nwalkers,burnin,nsteps,pos0,initial,params,m,t,lc_data_white,yerr_white,color)
    
        t0o,pero,rpo,inco,c1o,c2o=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(runwhite, [16, 50, 84], axis=0)))
        t0,per,rp,inc,c1,c2=t0o[0],pero[0],rpo[0],inco[0],c1o[0],c2o[0]
        u=[c1,c2]
        sma=(((per*24.*60.*60.)**2.*Grav*Ms)/(4*np.pi**2.))**(1./3.)
        sma=sma/Rs

        params,m=lc_func(t,t0,per,rp,sma,inc,ecc,w,u,limb_dark)
        paramsp,mp=lc_func(fulltime,t0,per,rp,sma,inc,ecc,w,u,limb_dark)
        
        fitlightcurve=m.light_curve(params)
        fitlightcurvep=mp.lightcurve(paramsp)
        
        residuals=(fitlightcurve-lc_data_white)*10**6.
        chi2=np.nansum(np.abs(residuals/10**6.)**2.)

        Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
        Output.write('{0} \n'.format('     -->> Best Fit Params'))
        Output.write('{0} {1} {2} {3} \n'.format('          t0  : ', t0,  np.round(t0o[1],5),np.round(t0o[2],5)))
        Output.write('{0} {1} {2} {3} \n'.format('          per : ', per, np.round(pero[1],10),np.round(pero[2],10)))
        Output.write('{0} {1} {2} {3} \n'.format('          rp  : ', rp,  np.round(rpo[1],5),np.round(rpo[2],5)))
        Output.write('{0} {1} \n'.format('          a   : ', sma))#, np.round(smao[1],5),np.round(smao[2],5)))                                      
        Output.write('{0} {1} {2} {3} \n'.format('          inc : ', inc, np.round(inco[1],5),np.round(inco[2],5)))
        Output.write('{0} {1} {2} {3} \n'.format('          c1  : ', c1,  np.round(c1o[1],5),np.round(c1o[2],5)))
        Output.write('{0} {1} {2} {3} \n'.format('          c2  : ', c2,  np.round(c2o[1],5),np.round(c2o[2],5)))
        Output.write('{0} {1} \n'.format('  chi-squared : ', chi2))
        Output.close()

        plt.figure()
#    plt.clf()                                                                                                                                  
        plt.plot(t,lc_data_white,'.',markersize=10,markeredgecolor='black',markerfacecolor='grey')
        plt.plot(fulltime,fitlightcurvep,'-',color='black')
        plt.ylim(0.96,1.01)
        plt.figtext(0.15,0.15,'$\chi^2$ = '+str(chi2))
#    plt.figtext(0.55,0.60, str(int(bin_wav[b]))+' $\AA$',fontsize=25,color=scal_m.to_rgba(bin_wav[b]))                                         
        plt.figtext(0.55,0.80, 'White Light', fontsize=25,color='grey')
        plt.savefig(SAVEPATH+'Fits_'+str(int(width))+'/Fit_Orbit_LC_white.png')
        plt.close()

        t0min=np.max([t0dn,t0-5*t0o[2]])
        t0max=np.min([t0up,t0+5*t0o[1]])
        rpmin=np.max([rpdn,rp-5*rpo[2]])
        rpmax=np.min([rpup,rp+5*rpo[1]])
    
        plt.clf()
        corner.corner(runwhite,labels=['t0','per','rp','inc','c1','c2'],truths=[t0,per,rp,inc,c1,c2],range=([t0min,t0max],[per-5*pero[2],per+5*pero[1]],[rpmin,rpmax],[inc-5*inco[2],inc+5*inco[1]],[c1-5*c1o[2],c1+5*c1o[1]],[c2-5*c2o[2],c2+5*c2o[1]]))
        plt.savefig(SAVEPATH+'Fits_'+str(int(width))+'/CornerPlot_white.png')
        plt.close()

        Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
        Output.write('{0}\n'.format(' '))
        Output.write('{0} {1} \n'.format('            TIME TO RUN: ', datetime.now() - startTime1))
        Output.write('{0}\n'.format(' '))
        Output.close()

        params=np.array([params.t0,params.per,params.rp,params.a,params.inc,params.u[0],params.u[1]])
        paramserr=np.array([[t0o[1],t0o[2]],[pero[1],pero[2]],[rpo[1],rpo[2]],[inco[1],inco[2]]])

        np.savez_compressed(SAVEPATH+'Fits_'+str(int(width))+'/LightCurve_fits_white.npz',results=runwhite,params=params,paramserr=paramserr,lightcurve_fit=fitlightcurvep,lcfitz=fitlightcurve,residuals=residuals)
    
    

    if color==1:
        white_fit=np.load(SAVEPATH+'Fits_'+str(int(width))+'/LightCurve_fits_white.npz')['params']
        t0,per,rp,sma,inc,c1,c2=white_fit
        
        for b in range(0,len(bin_ctr)):
            Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
            Output.write('{0} {1} {2}\n'.format('>>>>>>>>>> WAVELENGTH BIN: ' ,bin_ctr[b], ' <<<<<<<<<<'))
#        Output.write('{0} {1} \n'.format(' -->> WAVLENGTH CENTER = ', bin_ctr[b]))
            print('>>>>>>>>>> WAVELENGTH BIN: ' ,bin_ctr[b], ' <<<<<<<<<<')
    
    
            c1=q0_func(bin_ctr[b])
            c2=q1_func(bin_ctr[b])
            u=[c1,c2]
                                                                                                                          
            params,m=lc_func(t,t0,per,rp,sma,inc,ecc,w,u,limb_dark)
    
            initial=np.array([rp,c1,c2])
            ndim=len(initial)
    
            Output.write('{0} {1} \n'.format('     -->> Initial Guess: ', initial))
            Output.close()

            rp_arr=np.linspace(rpdn,rpup,10**6.)                                                                                              
            c1_arr=np.linspace(c1-0.2,c1+0.2,10**6.)
            c2_arr=np.linspace(c2-0.2,c2+0.2,10**6.)  

            pos0=np.empty([nwalkers,ndim])
            for i in range(nwalkers):
                pos0[i,:]=np.array([np.random.choice(rp_arr),np.random.choice(c1_arr),np.random.choice(c2_arr)])
        
            runlam=runmcmc(SAVEPATH,width,ndim,nwalkers,burnin,nsteps,pos0,initial,params,m,t,lc_data_binns[:,b],yerr_binns[:,b],color)
    
            rpo,c1o,c2o=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(runlam, [16, 50, 84], axis=0)))
            rp,c1,c2=rpo[0],c1o[0],c2o[0]
            u=[c1,c2]

            params,m=lc_func(t,t0,per,rp,sma,inc,ecc,w,u,limb_dark)
            paramsp,mp=lc_func(fulltime,t0,per,rp,sma,inc,ecc,w,u,limb_dark)
    
            fitlightcurve=m.light_curve(params)
            fitlightcurvep=mp.light_curve(paramsp)
        
            residuals=(fitlightcurve-lc_data_binns[:,b])*10**6.
            chi2=np.nansum(np.abs(residuals/10**6.)**2.)

            Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
            Output.write('{0} \n'.format('     -->> Best Fit Params'))
            Output.write('{0} {1} \n'.format('          t0  : ', t0))#,  np.round(t0o[1],5),np.round(t0o[2],5)))                                        
            Output.write('{0} {1} \n'.format('          per : ', per))#, np.round(pero[1],5),np.round(pero[2],5)))                                      
            Output.write('{0} {1} {2} {3} \n'.format('          rp  : ', rp,  np.round(rpo[1],5),np.round(rpo[2],5)))
            Output.write('{0} {1} \n'.format('          a   : ', sma))#, np.round(smao[1],5),np.round(smao[2],5)))                                      
            Output.write('{0} {1} \n'.format('          inc : ', inc))#, np.round(inco[1],5),np.round(inco[2],5)))                                      
            Output.write('{0} {1} {2} {3} \n'.format('          c1  : ', c1,  np.round(c1o[1],5),np.round(c1o[2],5)))
            Output.write('{0} {1} {2} {3} \n'.format('          c2  : ', c2,  np.round(c2o[1],5),np.round(c2o[2],5)))
            Output.write('{0} {1} \n'.format('  chi-squared : ', chi2))
            Output.close()

            plt.figure()
#    plt.clf()                                                                                                                                  
            plt.plot(t,lc_data_binns[:,b],'.',markersize=10,markeredgecolor='black',markerfacecolor=scal_m.to_rgba(bin_ctr[b]))
            plt.plot(fulltime,fitlightcurvep,'-',color='black')
            plt.ylim(0.96,1.01)
            plt.figtext(0.15,0.15,'$\chi^2$ = '+str(chi2))
            plt.figtext(0.55,0.80, str(int(bin_ctr[b]))+' $\AA$',fontsize=25,color=scal_m.to_rgba(bin_ctr[b]))                                         
#        plt.figtext(0.55,0.60, 'White Light', fontsize=25,color='grey')
            plt.savefig(SAVEPATH+'Fits_'+str(int(width))+'/Fit_Orbit_LC_'+str(int(bin_ctr[b]))+'.png')
            plt.close()

            rpmin=np.max([rpdn,rp-5*rpo[2]])
            rpmax=np.min([rpup,rp+5*rpo[1]])
    
            plt.clf()
            corner.corner(runlam,labels=['rp','c1','c2'],truths=[rp,c1,c2],range=([rpmin,rpmax],[c1-5*c1o[2],c1+5*c1o[1]],[c2-5*c2o[2],c2+5*c2o[1]]))
            plt.savefig(SAVEPATH+'Fits_'+str(int(width))+'/CornerPlot_'+str(int(bin_ctr[b]))+'.png')
            plt.close()

            Output=open(SAVEPATH+'Fits_'+str(int(width))+'/Progress'+str(int(color))+'.txt','a')
            Output.write('{0}\n'.format(' '))
            Output.write('{0} {1} \n'.format('            TIME TO RUN: ', datetime.now() - startTime1))
            Output.write('{0}\n'.format(' '))
            Output.close()

            params=np.array([params.t0,params.per,params.rp,params.a,params.inc,params.u[0],params.u[1]])
            paramserr=np.array([[rpo[1],rpo[2]],[c1o[1],c1o[2]],[c2o[1],c2o[2]]])

            np.savez_compressed(SAVEPATH+'Fits_'+str(int(width))+'/LightCurve_fits_'+str(int(bin_ctr[b]))+'.npz',results=runlam,params=params,paramserr=paramserr,lightcurve_fit=fitlightcurvep,lcfitz=fitlightcurve,residuals=residuals)