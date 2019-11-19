#####################################################                                                              
#####################################################                                                              
##        Fitting of lightcurves using SPOTROD     ##                                                          
##  ---------------------------------------------  ##                                                             
##  reads in .npz files from Lightcurve_Script.py  ##                                                              
##     See README.txt in Magellan/pipeline         ##                                                              
#####################################################                                                              
#####################################################  
import numpy as np
np.seterr(all='ignore')

from SystemCons_W52 import *
print name
#print midtime

import spotrod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import astropy
from astropy.time import Time

import scipy
from scipy.interpolate import interp1d

import emcee
import corner

from datetime import datetime
startTime = datetime.now()


####################################################   
path='/u/Exocomputer1/ermay/TransitModels/'
name=raw_input('INPUT TARGET NAME: ')
date=raw_input('INPUT OBSRVN DATE: ')


if name.find('Wasp52') != -1:
    from SystemCons_W52 import *
    if date.find('ut20160811') != -1:
        midtime=mid1  #Transit1                                                                                                   
    if date.find('ut20160922') != -1:
        midtime=mid2  #Transit2                                                                                                                        
if name.find('Wasp4') != -1:
    from SystemCons_W4 import *
if name.find('HatP26') != -1:
    from SystemCons_HP26 import *
print name
print midtime

time=np.genfromtxt(path+date+'/obs_times.txt',usecols=[0],dtype='datetime64')
n_exp=len(time)
mid_transit=np.datetime64(midtime)

td=(time-mid_transit)
t=np.linspace(0,1,n_exp)*0.0
for i in range(0,n_exp):
    t[i]=np.float((td[i])/np.timedelta64(1,'s'))/(60.*60.*24.)

lc_data=np.load(path+date+'/Calibrated_LC_data_bins_200.npz')['calibrated_data']
bin_wav=np.load(path+date+'/Calibrated_LC_data_bins_200.npz')['centers']
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
#                  SETUP BLOCK                     #                                                                        
####################################################                                                                        
                                                                                                                            
white_params=np.load(path+date+'/LightCurve_batman_fits.npz')['params']
white_errors=np.load(path+date+'/LightCurve_batman_fits.npz')['paramserr']

#orbital parameters                                                                                                         
rp=white_params[2]                       #[units of stellar radius]                                                         

per=white_params[1]                             #[period in days]                                                         
perdn=white_errors[1][0]
perup=white_errors[1][1]

sma=white_params[3]                       #[units of stellar radius]                                                        

inc=white_params[4]                       #[orbital inclination] 
                                                                            
incdn=white_errors[3][0]
incup=white_errors[3][1]

t0=white_params[0]                     #[time of center of transit [days]]                                                  
                                                                                                                          
t0dn=white_errors[0][0]
t0up=white_errors[0][1]


spot_params=np.load(path+date+'/LightCurve_batman_fits_spot.npz')['params']
spot_errors=np.load(path+date+'/LightCurve_batman_fits_spot.npz')['paramserr']

spotx=spot_params[0]
spoty=spot_params[1]
spotradius=spot_params[2]
spotcontrast=spot_params[3]


####################################################
#              LIMB DARKENING FUNCION              #
####################################################
# U B V R I J H K
wave=[3640,4450,5510,6580,8060,12200]#,16300,21900]                                                                          

q0_func=interp1d(wave,q0)
q1_func=interp1d(wave,q1)

def quadraticlimbdarkening(r,u1,u2):
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
#f = 2.0 * quadraticlimbdarkening(r, u1, u2) / n

# Calculate orbital elements.
#eta, xi = spotrod.elements(t, period, semimajoraxis, k, h)
#if not np.isfinite(np.mean(eta)):
#    Mean_Anomaly=(2*np.pi/period)*t
#    eta=semimajoraxis*np.cos(Mean_Anomaly)
#    xi=semimajoraxis*np.sin(Mean_Anomaly)
#planetx = impactparam*eta/semimajoraxis
#planety = -xi
#z = np.sqrt(np.power(planetx,2) + np.power(planety,2))

# Calculate planetangle array.
#planetangle = np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])])

#spotx=0.55492
#spoty=0.1832
#spotradius=0.06636
#spotcontrast=1.0

spotx=0.0
spoty=0.0
spotradius=0.0
spotcontrast=0.0


t_f=t-t0
smacm=(((per*24.*60.*60.)**2.*Grav*Ms)/(4*np.pi**2.))**(1./3.)
sma=smacm/Rs
MA=(2*np.pi/per)*t_f
eta=sma*np.cos(MA)
xi=sma*np.sin(MA)
b=sma*np.cos(inc*np.pi/180.)
planetx=b*eta/sma
planety=-xi
z=np.sqrt(planetx**2.+planety**2.)

def lightcurve(rp,c1,c2,spotcontrast):
    f=2.0*quadraticlimbdarkening(r,c1,c2)
    planetangle=np.array([spotrod.circleangle(r,rp,z[i]) for i in xrange(z.shape[0])])
    fitlightcurve=spotrod.integratetransit(planetx,planety,z,rp,r,f,np.array([spotx]),np.array([spoty]),np.array([spotradius]),np.array([spotcontrast]),planetangle)
    return fitlightcurve


####################################################
#     Prior, Likelihood, Posterior Functions       #
####################################################
def loggaus(mean,sig,param):
    return -np.log(np.sqrt(2*np.pi*sig**2.))-((param-mean)**2.)/(2*sig**2.)

def lnprior(theta):
    rp_f,c1_f,c2_f,spotcontrast=theta
    if spotcontrast<=1.0 and spotcontrast>=0.0 and rp_f>rpdn and rp_f<rpup and np.abs(c2_f-c2)<0.1 and np.abs((c2_f-c2)-(c1_f-c1))<0.05:
        return 0.0
    else:
        return -np.inf

def lnlike(theta,flux,err):
    rp_f,c1_f,c2_f,spotcontrast=theta
    fit=lightcurve(rp_f,c1_f,c2_f,spotcontrast)
    residuals=flux-fit
    return -0.5*(np.nansum((residuals/err)**2.))

def lnpost(theta,flux,err):
    prior=lnprior(theta)
    if not np.isfinite(prior):
        return -np.inf
    post=prior+lnlike(theta,flux,err)
    return post


####################################################
#u1_arr=np.array([np.nan,np.nan,np.nan,0.71365494,0.71362256,0.62765916,0.66887914,0.5895842,0.55162063,0.51973574,0.50610235,0.481512,0.41502468,np.nan,np.nan,0.31851643,0.29293669,0.39335547,0.16970518,0.31348718])

nwalkers=250
burnin=100
nsteps=1000

Output=open(path+date+'/Bins/Progress.txt','w')

for b in range(0,numbins):
    Output=open(path+date+'/Bins/Progress.txt','a')
    if np.isfinite(lc_data[0,b])==False:
        Output.write('{0} {1} {2}\n'.format('( skip bin ', b,')'))
        continue
    Output.write('{0} {1} {2}\n'.format('>>>>>>>>>> WAVELENGTH BIN NUMBER ' ,b, ' <<<<<<<<<<'))
    Output.write('{0} {1} \n'.format(' -->> WAVLENGTH CENTER = ', bin_wav[b]))
    
    c1 = q0_func(bin_wav[b])
    c2 = q1_func(bin_wav[b])
        
    initial=np.array([rp,c1,c2,spotcontrast])
    ndim=len(initial)
    
    rp_arr=np.linspace(rpdn,rpup,10**6.)
    c1_arr=np.linspace(c1-0.2,c1+0.2,10**6.)
    c2_arr=np.linspace(c2-0.2,c2+0.2,10**6.)
    sc_arr=np.linspace(0.0,1.0,10**6.)

    pos0=np.empty([nwalkers,ndim])
    for i in range(nwalkers):
        pos0[i,:]=np.array([np.random.choice(rp_arr),np.random.choice(c1_arr),np.random.choice(c2_arr),np.random.choice(sc_arr)])

    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnpost,a=2.0,args=(lc_data[:,b],10**-4))

    Output.write('{0} {1} \n'.format('     -->> Initial Guess: ', initial))
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
            Output=open(path+date+'/Bins/Progress.txt','a')
            Output.write('{0} {1} {2} \n'.format(("            {0:5.1%}".format(float(i+1) / nsteps)),'          ', datetime.now()))
            Output.close()

    Output=open(path+date+'/Bins/Progress.txt','a')
    Output.write('{0} {1} \n'.format('     -->> Mean Acceptance Fraction: ', np.mean(sampler.acceptance_fraction)))
    Output.close()

    samples=sampler.chain[:,:,:].reshape((-1,ndim))

    rpo,c1o,c2o,sco=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    #rpo,c1o,c2o=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    rp,c1,c2,spotcontrast=rpo[0],c1o[0],c2o[0],sco[0]
    #rp,c1,c2=rpo[0],c1o[0],c2o[0]
    u=[c1,c2]
    
    fitlightcurve=lightcurve(rp,c1,c2,spotcontrast)

    residuals=(fitlightcurve-lc_data[:,b])*10**6.
    chi2=np.nansum(np.abs(residuals/10**6.)**2.)
    
    Output=open(path+date+'/Bins/Progress.txt','a')
    Output.write('{0} \n'.format('     -->> Best Fit Params'))
    Output.write('{0} {1} \n'.format('          t0  : ', t0))#,  np.round(t0o[1],5),np.round(t0o[2],5)))
    Output.write('{0} {1} \n'.format('          per : ', per))#, np.round(pero[1],10),np.round(pero[2],10)))
    Output.write('{0} {1} {2} {3} \n'.format('          rp  : ', rp,  np.round(rpo[1],5),np.round(rpo[2],5)))
    Output.write('{0} {1} \n'.format('          a   : ', sma))#, np.round(smao[1],5),np.round(smao[2],5)))                                 
    Output.write('{0} {1} \n'.format('          inc : ', inc))#, np.round(inco[1],5),np.round(inco[2],5)))
    Output.write('{0} {1} {2} {3} \n'.format('          c1  : ', c1,  np.round(c1o[1],5),np.round(c1o[2],5)))
    Output.write('{0} {1} {2} {3} \n'.format('          c2  : ', c2,  np.round(c2o[1],5),np.round(c2o[2],5)))
    Output.write('{0} {1} {2} {3} \n'.format('          sc  : ', spotcontrast,  np.round(sco[1],5),np.round(sco[2],5)))
    Output.write('{0} {1} \n'.format('  chi-squared : ', chi2))
    Output.close()

    plt.figure()
#    plt.clf()                                                                                                                             
    plt.plot(t,lc_data[:,b],'.',markersize=10,markeredgecolor='black',markerfacecolor=scal_m.to_rgba(bin_wav[b]))
    plt.plot(t,fitlightcurve,'-',color='black')
    plt.ylim(0.96,1.005)
    plt.figtext(0.15,0.15,'$\chi^2$ = '+str(chi2))
    plt.figtext(0.15,0.60, str(int(bin_wav[b]))+' $\AA$',fontsize=25,color=scal_m.to_rgba(bin_wav[b]))                                    
#    plt.figtext(0.55,0.60, 'White Light', fontsize=25,color=scal_m.to_rgba(bin_wav[b]))
    plt.savefig(path+date+'/Bins/'+str(int(b))+'_Fit_Orbit_LC.png')
    plt.close()


    #t0min=np.max([t0dn,t0-5*t0o[2]])
    #t0max=np.min([t0up,t0+5*t0o[1]])
    rpmin=np.max([rpdn,rp-5*rpo[2]])
    rpmax=np.min([rpup,rp+5*rpo[1]])
    scmin=np.max([0.0,spotcontrast-5*sco[2]])
    scmax=np.min([1.0,spotcontrast+5*sco[1]])

    plt.clf()
    corner.corner(samples,labels=['rp','c1','c2','sc'],truths=[rp,c1,c2,spotcontrast],range=([rpmin,rpmax],[c1-5*c1o[2],c1+5*c1o[1]],[c2-5*c2o[2],c2+5*c2o[1]],[scmin,scmax]))
    #corner.corner(samples,labels=['rp','c1','c2'],truths=[rp,c1,c2],range=([rpmin,rpmax],[c1-5*c1o[2],c1+5*c1o[1]],[c2-5*c2o[2],c2+5*c2o[1]]))
    plt.savefig(path+date+'/Bins/'+str(int(b))+'_CornerPlot.png')
    plt.close()

    Output=open(path+date+'/Bins/Progress.txt','a')
    Output.write('{0}\n'.format(' '))
    Output.write('{0} {1} \n'.format('            TIME TO RUN: ', datetime.now() - startTime))
    Output.write('{0}\n'.format(' '))
    Output.close()

    params=np.array([rp,c1,c2,spotcontrast])
    paramserr=np.array([[rpo[1],rpo[2]],[c1o[1],c1o[2]],[c2o[1],c2o[2]],[sco[1],sco[2]]])

    np.savez_compressed(path+date+'/Bins/'+str(int(b))+'_LightCurve_batman_fits.npz',results=samples,params=params,paramserr=paramserr,lightcurve_fit=fitlightcurve,residuals=residuals)
