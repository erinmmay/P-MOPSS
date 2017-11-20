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

path='/Users/ermay/Desktop/'
name='Wasp52'

times=np.genfromtxt(path+name+'/obs_times.txt',usecols=[0],dtype='string')
time=Time(times,format='isot',scale='utc')
#midtime=input('Enter Transit Midpoint: ')
midtime=['2016-09-22T04:44:00']
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

### for now...###
b=0
flux=lc_data[:,b]


#transit params for SPOTROD #
period = 1.9#84
periodhour = 24.0*period
rp = 0.169
#semimajoraxis = 7.253184
semimajoraxis=6.975
ecc=0.0
pericenter=0.0
k = ecc*np.cos(pericenter)
h = ecc*np.sin(pericenter)
impactparam = 0.656
u1 = q0_func(bin_wav[b])
u2 = q1_func(bin_wav[b])

t_off=-0.01
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
planetangle = np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])])

#spotx = 0.5
#spoty = 0.19
#spotradius = 0.09
spotx=0.55492
spoty=0.1832
spotradius=0.06636
spotcontrast=1.0


#contrast_arr=np.linspace(0.05,0.95,19)
#spotcontrast = 0.27
#fnum=14

fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
fit_params=[planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle]

chi2=np.sum((flux-fitlightcurve)**2.)

plt.figure(2)
plt.clf()
plt.plot(t,flux,'b.')
plt.plot(t,fitlightcurve,'k-')
#plt.figtext(0.15,0.70,'CONTRAST LEVEL ',fontsize=20,color='blue')
#plt.figtext(0.15,0.65,'          '+str(spotcontrast),fontsize=20,color='blue')
plt.figtext(0.15,0.60,'Spotx = '+str(np.round(spotx,4)),fontsize=15)
plt.figtext(0.15,0.55,'Spoty = '+str(np.round(spoty,4)),fontsize=15)
plt.figtext(0.15,0.50,'Spotr = '+str(np.round(spotradius,4)),fontsize=15)
plt.figtext(0.15,0.45,'Spotc = '+str(np.round(spotcontrast,4)),fontsize=15)
plt.figtext(0.15,0.35,'$\chi^2$ = '+str(chi2))
plt.savefig('/Users/ermay/Desktop/spot_LC_0.png')
plt.close()
####################################################
#     Prior, Likelihood, Posterior Functions       #
####################################################
def lnprior(theta):
    diff=1.5
    if theta[0]<1.0 and theta[0]>-1.0 and theta[1]<1.0 and theta[1]>-1.0 and theta[2]<0.5 and theta[2]>0.01 and theta[3]>1.0:
        return 0.0
    else:
        return -np.inf

def lnlike(theta,fit_params,flux,err):
    fit_params[6],fit_params[7],fit_params[8],fit_params[9]=theta[0],theta[1],theta[2],theta[3]
    planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle=fit_params
    fit=spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    residuals=flux-fit
    return -0.5*(np.nansum((residuals/err)**2.))

def lnpost(theta,fit_params,flux,err):
    prior=lnprior(theta)
    if not np.isfinite(prior):
        return -np.inf
    post=prior+lnlike(theta,fit_params,flux,err)
    return post

initial_guess=[spotx,spoty,spotradius,spotcontrast]
#initial_guess=[spotx,spotcontrast]
ndim=len(initial_guess)
nwalkers=5000
burnin=1000
nsteps=15000

y_arr=np.linspace(-1.0,1.0,10**6.)
x_arr=np.linspace(-1.0,1.0,10**6.)
r_arr=np.linspace(0.01,0.3,10**6.)
f_arr=np.linspace(1.0,1.5,10**6.)

pos0=np.empty([nwalkers,ndim])
for i in range(nwalkers):
    pos0[i,:]=np.array([np.random.choice(x_arr),np.random.choice(y_arr),np.random.choice(r_arr),np.random.choice(f_arr)])
#    pos0[i,:]=np.array([np.random.choice(x_arr),np.random.choice(f_arr)])#,np.random.choice(r_arr),np.random.choice(f_arr)])


sampler=emcee.EnsembleSampler(nwalkers,ndim,lnpost,a=0.1,args=(fit_params,lc_data[:,b],10**-4))
#print '>>>>> CONTRAST LEVEL ',spotcontrast, ' <<<<<'
print '     -->> Initial Guess: ', initial_guess
#print '                u= ', [u1,u2]
print '     -->> Running Burn-in...'

p0,_,_=sampler.run_mcmc(pos0,burnin)

print '     -->> Running Chain...'
for i, result in enumerate(sampler.sample(p0,iterations=nsteps)):
    if (i+1)%(nsteps/10) ==0:
        print("            {0:5.1%}".format(float(i+1) / nsteps)),'          ', datetime.now()
    
print '     -->> Mean Acceptance Fraction: ', np.mean(sampler.acceptance_fraction)

samples=sampler.chain[:,:,:].reshape((-1,ndim))

best_spotx,best_spoty,best_spotradius,best_contrast=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
fit_params[6],fit_params[7],fit_params[8],fit_params[9]=best_spotx[0],best_spoty[0],best_spotradius[0],best_contrast[0]
#best_spotx,best_contrast=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
#fit_params[6],fit_params[9]=best_spotx[0],best_spoty[0],best_spotradius[0],best_contrast[0]

planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle=fit_params
fitlightcurve=spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    
chi2=np.sum((flux-fitlightcurve)**2.)
#print '     -->> FIT results FOR contrast level ',spotcontrast,' (',np.round(5000-((spotcontrast**(0.25))*5000),2),'K ) : '
#print '             Planet Radius:', best_rp
print '                    Spot x:', np.round(best_spotx,5)
print '                    Spot y:', np.round(best_spoty,5)
print '               Spot Radius:', np.round(best_spotradius,5)
print '             Spot Contrast:', np.round(best_contrast,5)
print ' '
print '          10^4xChi-Squared:', 10000*chi2
print ' '

plt.figure(2)
plt.clf()
plt.plot(t,flux,'b.')
plt.plot(t,fitlightcurve,'k-')
#plt.figtext(0.15,0.70,'CONTRAST LEVEL ',fontsize=20,color='blue')
#plt.figtext(0.15,0.65,'          '+str(spotcontrast),fontsize=20,color='blue')
plt.figtext(0.15,0.60,'Spotx = '+str(np.round(spotx,4)),fontsize=15)
plt.figtext(0.15,0.55,'Spoty = '+str(np.round(spoty,4)),fontsize=15)
plt.figtext(0.15,0.50,'Spotr = '+str(np.round(spotradius,4)),fontsize=15)
plt.figtext(0.15,0.45,'Spotc = '+str(np.round(spotcontrast,4)),fontsize=15)
plt.figtext(0.15,0.35,'$\chi^2$ = '+str(chi2))
plt.savefig('/Users/ermay/Desktop/brightspot_LC.png')
plt.close()

eran=5.
sx_dn=np.max([-1.0,best_spotx[0]-eran*best_spotx[1]])
sx_up=np.min([1.0,best_spotx[0]+eran*best_spotx[2]])
sy_dn=np.max([-1.0,best_spoty[0]-eran*best_spoty[1]])
sy_up=np.min([1.0,best_spoty[0]+eran*best_spoty[2]])
sr_dn=np.max([0.01,best_spotradius[0]-eran*best_spotradius[1]])
sr_up=np.min([0.3,best_spotradius[0]+eran*best_spotradius[2]])
sc_dn=np.max([0.01,best_contrast[0]-eran*best_contrast[1]])
sc_up=np.min([0.5,best_contrast[0]+eran*best_contrast[2]])
plt.clf()
corner.corner(samples,labels=['x','y','r','f'],truths=[best_spotx[0],best_spoty[0],best_spotradius[0],best_contrast[0]],range=[(sx_dn,sx_up),(sy_dn,sy_up),(sr_dn,sr_up),(sc_dn,sc_up)])
plt.savefig('/Users/ermay/Desktop/brightspot_corner.png')
plt.close()

print ' '
print '            TIME TO RUN: ', datetime.now() - startTime
print ' '


X,Y=np.meshgrid(np.linspace(-1.0,1.0,10**3.),np.linspace(-1.0,1.0,10**3.))
circle=X**2.+Y**2.

spot=(X-spotx)**2.+(Y-spoty)**2.
limb=quadraticlimbdarkening(r, u1, u2)

norm=matplotlib.colors.Normalize(vmin=0.1 ,vmax=spotcontrast)
#norm=matplotlib.colors.BoundaryNorm(boundaries=np.linspace(0.0,1.0,100),ncolors=100*len(limb))
#colors=matplotlib.cm.RdYlBu_r
#colors=matplotlib.cm.Greys
colors=matplotlib.cm.afmhot
scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
scal_m.set_array([])
    

plt.figure(figsize=(9.,9.))
plt.contour(Y,X,circle,levels=[1.0],linewidths=1.,colors='black',zorder=0)
#for i in range(0,len(r)):
#    plt.contour(Y,X,circle,levels=[r[i]**2.],colors='grey',alpha=1.-limb[i])
plt.contour(Y,X,circle,levels=rsq,colors=scal_m.to_rgba(limb),zorder=1) 
plt.contour(Y,X,spot,levels=[spotradius**2.],linewidths=0.25,colors='black',zorder=3)
plt.contourf(Y,X,spot,levels=[0,spotradius**2],colors=scal_m.to_rgba([spotcontrast]),alpha=0.75,zorder=2)
#plt.contour(Y,X,spot,levels=[spotradius**2.],linewidths=0.5,colors='black')
p#lt.contourf(Y,X,spot,levels=[0,spotradius**2],colors='grey',alpha=1.-spotcontrast)
plt.plot(planety,planetx,'--',linewidth=0.5,color='skyblue')
plt.plot(planety,planetx+rp,'-',linewidth=2,color='skyblue')
plt.plot(planety,planetx-rp,'-',linewidth=2,color='skyblue')
plt.ylim(-1.5,1.5)
plt.xlim(-1.5,1.5)
#plt.show()
plt.savefig('/Users/ermay/Desktop/brightspot_loc.png')
plt.close()
