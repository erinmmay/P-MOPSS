import numpy as np
import spotrod

import matplotlib.pyplot as plt

import astropy
from astropy.time import Time

import scipy
from scipy.interpolate import interp1d

import emcee
import corner

from datetime import datetime
startTime = datetime.now()

#path='/Users/ermay/Desktop/'
path='/Volumes/ermay_ext/Magellan/python_pipeline/'
name='Wasp52'

times=np.genfromtxt(path+name+'/obs_times.txt',usecols=[0],dtype='string')
time=Time(times,format='isot',scale='utc')
#midtime=input('Enter Transit Midpoint: ')
midtime=['2016-08-11T04:51:00']
mid_transit=Time(midtime,format='isot',scale='utc')
time-=mid_transit
t=np.linspace(time[0].value,time[-1].value,len(time))

n_exp=len(time)

lc_data=np.load(path+name+'/Calibrated_LC_data.npz')['calibrated_data']
bin_wav=np.load(path+name+'/Calibrated_LC_data.npz')['centers']

for i in range(0,n_exp):
    for j in range(0,len(bin_wav)):
        if lc_data[i,j]>1.1 or lc_data[i,j]<0.96:
            lc_data[i,j]=np.nan

####################################################
#              LIMB DARKENING FUNCION              #
####################################################
# U B V R I J H K

wave=[3640,4450,5510,6580,8060,12200]#,16300,21900]

### WASP 4 ####
q0=[0.9264,0.7208,0.5216,0.4191,0.3307,0.1767]#,0.0590,0.0634]
q1=[-0.0652,0.1033,0.2184,0.2481,0.2514,0.2922]#,0.3503,0.2890]

### WASP 52 ####
#q0=[1.0666,0.8661,0.6538,0.5240,0.4100,0.2315]#,0.0750,0.0767]
#q1=[-0.2171,-0.0193,0.1211,0.1803,0.2043,0.2752]#,0.3770,0.3097]

### GJ3470 ####
#q0=[0.4474,0.4271,0.4267,0.4233,0.2847,0.0232]#,0.0009,-0.0015]
#q1=[0.3377,0.3783,0.3504,0.3003,0.3611,0.3779]#,0.3109,0.2642]

q0_func=interp1d(wave,q0)
q1_func=interp1d(wave,q1)

### for now...###
b=0
flux=lc_data[:,b]


#transit params for SPOTROD #
period = 1.749
periodhour = 24.0*period
rp = 0.1598
semimajoraxis = 7.404
ecc=0.0
pericenter=0.0
k = ecc*np.cos(pericenter)
h = ecc*np.sin(pericenter)
impactparam = semimajoraxis*np.cos(85.35*np.pi/180.)
u1 = q0_func(bin_wav[b])
u2 = q1_func(bin_wav[b])

#u1=0.65
#u2=0.12

t_off=-0.0167
t-=t_off

per_dn=1.0
per_up=2.5

sma_dn=6.5
sma_up=8.5

rad_dn=0.14
rad_up=0.19




# limbdarkening law #
def quadraticlimbdarkening(r, u1, u2):
  answer = np.zeros_like(r)
  mask = (r<=1.0)
  oneminusmu = 1.0 - np.sqrt(1.0 - np.power(r[mask],2))
  answer[mask] = 1.0 - u1 * oneminusmu - u2 * np.power(oneminusmu,2)
  return answer

# Initialize spotrod.
# Number of intergration rings.
n = 1000

# Midpoint rule for integration.
# Integration annulii radii.
r = np.linspace(1.0/(2*n), 1.0-1.0/(2*n), n)
# Weights: 2.0 times limb darkening times width of integration annulii.
f = 2.0 * quadraticlimbdarkening(r, u1, u2) / n


#spotx = 0.5
#spoty = 0.2
#spotradius = 0.1

#contrast_arr=np.linspace(0.05,0.95,19)
#spotcontrast = 0.25
#fnum=14

spotx=0.0
spoty=0.0
spotradius=0.0
spotcontrast=1.0

def lightcurve(fitparams,period,sma):
    # Calculate orbital elements.
    f,rp=fit_params
#    eta, xi = spotrod.elements(t, period, semimajoraxis, k, h)
#    if not np.isfinite(np.mean(eta)):
    Mean_Anomaly=(2*np.pi/period)*t
    eta=semimajoraxis*np.cos(Mean_Anomaly)
    xi=semimajoraxis*np.sin(Mean_Anomaly)
    impactparam = semimajoraxis*np.cos(85.35*np.pi/180.)
    planetx = impactparam*eta/semimajoraxis
    planety = -xi
    z = np.sqrt(np.power(planetx,2) + np.power(planety,2))
    # Calculate planetangle array.
    planetangle = np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])])
    fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    return fitlightcurve
    

####################################################
#     Prior, Likelihood, Posterior Functions       #
####################################################
def lnprior(theta):
    diff=1.5
#    if theta[0]<0.5 and theta[0]>-0.5 and theta[1]<0.5 and theta[1]>-0.5 and theta[2]<0.2 and theta[2]>0.01:
#        return 0.0
#    else:
#        return -np.inf
    return 0.0

def lnlike(theta,fit_params,flux,err):
#    fit_params[6],fit_params[7],fit_params[8]=theta[0],theta[1],theta[2]
    fit_params[1]=theta[2]
#    planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle=fit_params
    f,rp=fit_params
#    fit=spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    fit=lightcurve(fit_params,theta[0],theta[1])
    residuals=flux-fit
    return -0.5*(np.nansum((residuals/err)**2.))

def lnpost(theta,fit_params,flux,err):
    prior=lnprior(theta)
    if not np.isfinite(prior):
        return -np.inf
    post=prior+lnlike(theta,fit_params,flux,err)
    return post

####################################################

for c in range(0,1):
#    spotcontrast=contrast_arr[c]
#    fnum=len(contrast_arr)-c
    
    spotx = 0.0
    spoty = 0.0
    spotradius = 0.0
    spotcontrast=1.0

#    fit_params=[planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle]
    fit_params=[f,rp]
    fitlightcurve=lightcurve(fit_params,period,semimajoraxis)

    chi2=np.sum((flux-fitlightcurve)**2.)

    plt.figure(1)
    plt.clf()
    plt.plot(t,flux,'b.')
    plt.plot(t,fitlightcurve,'k-')
    plt.ylim(0.96,1.005)
#    plt.figtext(0.15,0.70,'CONTRAST LEVEL ',fontsize=20,color='blue')
#    plt.figtext(0.15,0.65,'          '+str(spotcontrast),fontsize=20,color='blue')
    plt.figtext(0.15,0.15,'$\chi^2$ = '+str(chi2))
    plt.savefig(path+name+'/Fit_Orbit_origLC.png')
    #plt.show(block=False)
    #plt.pause(2.0)
    plt.close()

#    initial_guess=[spotx,spoty,spotradius]
    initial_guess=[period,semimajoraxis,rp]
    ndim=len(initial_guess)
    nwalkers=500
    burnin=100
    nsteps=1000

    #noise=10**-1

#    y_arr=np.linspace(-0.5,0.5,10**6.)
#    x_arr=np.linspace(-0.5,0.5,10**6.)
#    r_arr=np.linspace(0.01,0.2,10**6.)
    #f_arr=np.linspace(0.0,1.0,10**6.)
    p_arr=np.linspace(per_dn,per_up,10**6.)
    a_arr=np.linspace(sma_dn,sma_up,10**6.)
    r_arr=np.linspace(rad_dn,rad_up,10**6.)
    


    #pos0=[initial_guess+noise*np.random.rand(ndim) for i in range(nwalkers)]
    pos0=np.empty([nwalkers,ndim])
    for i in range(nwalkers):
        pos0[i,:]=np.array([np.random.choice(p_arr),np.random.choice(a_arr),np.random.choice(r_arr)])

    #plt.clf()
    #corner.corner(pos0,labels=['x','y','r','f'])
    #plt.show()

    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnpost,a=2.0,args=(fit_params,lc_data[:,b],10**-4))
#    print '>>>>> CONTRAST LEVEL ',spotcontrast, ' <<<<<'
    print '     -->> Initial Guess: ', initial_guess
    print '                u= ', [u1,u2]
    print '     -->> Running Burn-in...'

    p0,_,_=sampler.run_mcmc(pos0,burnin)

    print '     -->> Running Chain...'
    for i, result in enumerate(sampler.sample(p0,iterations=nsteps)):
        if (i+1)%(nsteps/10) ==0:
            print("            {0:5.1%}".format(float(i+1) / nsteps)),'          ', datetime.now()
    
    print '     -->> Mean Acceptance Fraction: ', np.mean(sampler.acceptance_fraction)

    samples=sampler.chain[:,:,:].reshape((-1,ndim))

#    best_spotx,best_spoty,best_spotradius=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
#    fit_params[6],fit_params[7],fit_params[8]=best_spotx[0],best_spoty[0],best_spotradius[0]
    best_period,best_sma,best_pradius=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    period,semimajoraxis,fit_params[1]=best_period[0],best_sma[0],best_pradius[0]

#    planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle=fit_params
    f,rp=fit_params
#    fitlightcurve=spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    fitlightcurve=lightcurve(fit_params,period,semimajoraxis)    
    
    chi2=np.sum((flux-fitlightcurve)**2.)
#    print '     -->> FIT results FOR contrast level ',spotcontrast,' (',np.round(5000-((spotcontrast**(0.25))*5000),2),'K ) : '
    print '             Planet Radius:', best_pradius
    print '                    Period:', best_period
    print '           Semi-Major Axis:', best_sma
#    print '                    Spot x:', np.round(best_spotx,5)
#    print '                    Spot y:', np.round(best_spoty,5)
#    print '               Spot Radius:', np.round(best_spotradius,5)
#    print '             Spot Contrast:', spotcontrast
#    print ' '
    print '          10^4xChi-Squared:', 10000*chi2
#    print ' '


    plt.figure(2)
    plt.clf()
    plt.plot(t,flux,'b.')
    plt.plot(t,fitlightcurve,'k-')
    plt.ylim(0.96,1.005)
#    plt.figtext(0.15,0.70,'CONTRAST LEVEL ',fontsize=20,color='blue')
#    plt.figtext(0.15,0.65,'          '+str(spotcontrast),fontsize=20,color='blue')
#    plt.figtext(0.15,0.60,'Spotx = '+str(np.round(spotx,4)),fontsize=15)
#    plt.figtext(0.15,0.55,'Spoty = '+str(np.round(spoty,4)),fontsize=15)
#    plt.figtext(0.15,0.50,'Spotr = '+str(np.round(spotradius,4)),fontsize=15)
    plt.figtext(0.15,0.15,'$\chi^2$ = '+str(chi2))
    plt.savefig(path+name+'/Fit_Orbit_LC.png')
    plt.close()

    eran=5.

#    p_arr=np.linspace(0.8,1.8,10**6.)
#    a_arr=np.linspace(4.0,7.0,10**6.)
#    r_arr=np.linspace(0.14,0.19,10**6.)
#    sx_dn=np.max([-0.5,best_spotx[0]-eran*best_spotx[1]])
#    sx_up=np.min([0.5,best_spotx[0]+eran*best_spotx[2]])
#    sy_dn=np.max([-0.5,best_spoty[0]-eran*best_spoty[1]])
#    sy_up=np.min([0.5,best_spoty[0]+eran*best_spoty[2]])
#    sr_dn=np.max([0.01,best_spotradius[0]-eran*best_spotradius[1]])
#    sr_up=np.min([0.2,best_spotradius[0]+eran*best_spotradius[2]])
    per_up=np.min([per_up,best_period[0]+eran*best_period[2]])
    per_dn=np.max([per_dn,best_period[0]-eran*best_period[2]])
    sma_up=np.min([sma_up,best_sma[0]+eran*best_sma[2]])
    sma_dn=np.max([sma_dn,best_sma[0]-eran*best_sma[2]])
    rad_up=np.min([rad_up,best_pradius[0]+eran*best_pradius[2]])
    rad_dn=np.max([rad_dn,best_pradius[0]-eran*best_pradius[2]])
    plt.clf()
#    corner.corner(samples,labels=['x','y','r'],truths=[best_spotx[0],best_spoty[0],best_spotradius[0]],range=[(sx_dn,sx_up),(sy_dn,sy_up),(sr_dn,sr_up)])
    corner.corner(samples,labels=['per','sma','r'],truths=[best_period[0],best_sma[0],best_pradius[0]],range=[(per_dn,per_up),(sma_dn,sma_up),(rad_dn,rad_up)])
    plt.savefig(path+name+'/Fit_Orbit_Corner.png')
    plt.close()

    print ' '
    print '            TIME TO RUN: ', datetime.now() - startTime
    print ' '

