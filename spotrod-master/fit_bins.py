import numpy as np
import spotrod

import matplotlib
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
#name='Wasp52'
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
numbins=len(bin_wav)

for i in range(0,n_exp):
    for j in range(0,len(bin_wav)):
        if lc_data[i,j]>1.1 or lc_data[i,j]<0.96:
            lc_data[i,j]=np.nan

####################################################
norm=matplotlib.colors.Normalize(vmin=np.min(bin_wav),vmax=np.max(bin_wav))
#colors=matplotlib.cm.RdYlBu_r
colors=matplotlib.cm.Spectral_r
scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
scal_m.set_array([])

####################################################
#              LIMB DARKENING FUNCION              #
####################################################
# U B V R I J H K

wave=[3640,4450,5510,6580,8060,12200]#,16300,21900]

### WASP 4 ####
#q0=[0.9264,0.7208,0.5216,0.4191,0.3307,0.1767]#,0.0590,0.0634]
#q1=[-0.0652,0.1033,0.2184,0.2481,0.2514,0.2922]#,0.3503,0.2890]

### WASP 52 ####
q0=[1.0666,0.8661,0.6538,0.5240,0.4100,0.2315]#,0.0750,0.0767]
q1=[-0.2171,-0.0193,0.1211,0.1803,0.2043,0.2752]#,0.3770,0.3097]

### GJ3470 ####
#q0=[0.4474,0.4271,0.4267,0.4233,0.2847,0.0232]#,0.0009,-0.0015]
#q1=[0.3377,0.3783,0.3504,0.3003,0.3611,0.3779]#,0.3109,0.2642]

q0_func=interp1d(wave,q0)
q1_func=interp1d(wave,q1)

#transit params for SPOTROD #
period = 1.94#840301
periodhour = 24.0*period
rp = 0.167
#semimajoraxis = 7.253184
semimajoraxis=7.510
ecc=0.0
pericenter=0.0
k = ecc*np.cos(pericenter)
h = ecc*np.sin(pericenter)
impactparam = semimajoraxis*np.cos(85.35*np.pi/180.)
u1 = q0_func(bin_wav[0])
u2 = q1_func(bin_wav[0])

t_off=-0.0167
t-=t_off


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

# Calculate orbital elements.
eta, xi = spotrod.elements(t, period, semimajoraxis, k, h)
if not np.isfinite(np.mean(eta)):
    Mean_Anomaly=(2*np.pi/period)*t
    eta=semimajoraxis*np.cos(Mean_Anomaly)
    xi=semimajoraxis*np.sin(Mean_Anomaly)
planetx = impactparam*eta/semimajoraxis
planety = -xi
z = np.sqrt(np.power(planetx,2) + np.power(planety,2))

# Calculate planetangle array.
#planetangle = np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])])


#spotx = 0.5
#spoty = 0.2
#spotradius = 0.1

#contrast_arr=np.linspace(0.05,0.95,19)
#spotcontrast = 0.25
#fnum=14

#spotx=0.55492
#spoty=0.1832
#spotradius=0.06636
#spotcontrast=1.0
spotx=0.0
spoty=0.0
spotradius=0.0
spotcontrast=1.0


fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])]))
fit_params=[planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast]
    

####################################################
#     Prior, Likelihood, Posterior Functions       #
####################################################
def lnprior(theta):
    diff=1.5
    if theta[0]<0.19 and theta[0]>0.14:
        return 0.0
    else:
        return -np.inf

def lnlike(theta,fit_params,flux,err):
    fit_params[3]=theta[0]
    planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast=fit_params
    fit=spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])]))
    residuals=flux-fit
    return -0.5*(np.nansum((residuals/err)**2.))

def lnpost(theta,fit_params,flux,err):
    prior=lnprior(theta)
    if not np.isfinite(prior):
        return -np.inf
    post=prior+lnlike(theta,fit_params,flux,err)
    return post

####################################################

for b in range(0,numbins):
    if np.isfinite(lc_data[0,b])==False:
        print '( skip bin ', b,')'
        continue
    u1 = q0_func(bin_wav[b])
    u2 = q1_func(bin_wav[b])
    f = 2.0 * quadraticlimbdarkening(r, u1, u2) / n
    
    rp=0.167
#    spotcontrast=0.2348

    fit_params=[planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast]
    fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])]))

    plt.figure(1)
    plt.clf()
    plt.plot(t,fitlightcurve,color=scal_m.to_rgba(bin_wav[b]),linewidth=2.0,alpha=0.75)
    plt.plot(t,lc_data[:,b],'k.')
    plt.figtext(0.65,0.70,bin_wav[b],fontsize=25,color=scal_m.to_rgba(bin_wav[b]),fontweight='heavy')
    plt.ylim(0.965,1.005)
    plt.show(block=False)
    plt.pause(5.0)
    plt.close()

    initial_guess=[rp]
    ndim=len(initial_guess)
    nwalkers=100
    burnin=50
    nsteps=300
    
#    f_arr=np.linspace(0.05,1.0,10**6.)
    r_arr=np.linspace(0.14,0.19,10**6.)
    
    pos0=np.empty([nwalkers,ndim])
    for i in range(nwalkers):
        pos0[i,:]=np.array([np.random.choice(r_arr)])

    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnpost,a=2.0,args=(fit_params,lc_data[:,b],10**-4))
    print '>>>>>>>>>> WAVELENGTH BIN NUMBER ' ,b, ' <<<<<<<<<<'
    print ' -->> WAVLENGTH CENTER = ', bin_wav[b] 
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

    best_pradius=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    fit_params[3]=best_pradius[0][0]
    

    planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast=fit_params
    fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]),np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])]))
   
    chi2=np.nansum((lc_data[:,b]-fitlightcurve)**2.)
    residuals=lc_data[:,b]-fitlightcurve
    print '             Planet Radius:', best_pradius
 #   print '             Spot Contrast:', best_contrast
    print ' '
    print '          10^4xChi-Squared:', 10000*chi2
    print ' '


    plt.figure(2)
    plt.clf()
    plt.plot(t,fitlightcurve,color=scal_m.to_rgba(bin_wav[b]),linewidth=2.0,alpha=0.75)
    plt.plot(t,lc_data[:,b],'k.')
    plt.figtext(0.65,0.70,bin_wav[b],fontsize=25,color=scal_m.to_rgba(bin_wav[b]),fontweight='heavy')
#    plt.figtext(0.15,0.65,'contrast='+str(np.round(spotcontrast,4)),fontsize=20,color='black')
    plt.figtext(0.65,0.60,'pradius ='+str(np.round(rp,4)),fontsize=20,color='black')
    plt.figtext(0.65,0.55,'$\chi^2$ = '+str(np.round(chi2,8)),color='black')
    plt.ylim(0.965,1.005)
    plt.savefig(path+name+'/'+str(int(b))+'_Fit_LC.png')
    plt.close()

    eran=5.

#    sc_dn=np.max([0.10,best_contrast[0]-eran*best_contrast[1]])
#    sc_up=np.min([1.0,best_contrast[0]+eran*best_contrast[2]])
    rad_up=np.min([0.19,best_pradius[0][0]+eran*best_pradius[0][2]])
    rad_dn=np.max([0.14,best_pradius[0][0]-eran*best_pradius[0][2]])
    plt.clf()
    corner.corner(samples,labels=['rp'],truths=[best_pradius[0][0]],range=[(rad_dn,rad_up)])
#    corner.corner(samples,labels=['f','rp'],truths=[best_contrast[0],best_pradius[0]],range=[(0.01,1.0),(0.14,0.19)])
    plt.savefig(path+name+'/'+str(int(b))+'_Fit_Corner.png')
    plt.close()

    print ' '
    print '            TIME TO RUN: ', datetime.now() - startTime
    print ' '

    np.savez_compressed(path+name+'/'+str(int(b))+'_LightCurve_fits.npz',results=samples,fit_params=fit_params,lightcurve_fit=fitlightcurve,residuals=residuals) 
