import numpy as np
import spotrod

import matplotlib.pyplot as plt

import astropy
from astropy.time import Time

import scipy
from scipy.interpolate import interp1d

q0=[0.9264,0.7208,0.5216,0.4191,0.3307,0.1767]#,0.0590,0.0634]
q1=[-0.0652,0.1033,0.2184,0.2481,0.2514,0.2922]#,0.3503,0.2890]

t=np.linspace(-0.1,0.1,1000)

period = 1.749
periodhour = 24.0*period
rp = 0.1598
semimajoraxis = 7.404
ecc=0.0
pericenter=0.0
k = ecc*np.cos(pericenter)
h = ecc*np.sin(pericenter)
impactparam = semimajoraxis*np.cos(85.35*np.pi/180.)
u1 = q0[0]
u2 = q1[0]

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

spotx1=1.0
spoty1=0.0
spotradius1=0.4
spotcontrast1=0.1

spotx2=spotx1
spoty2=spoty1
spotradius2=0.2
spotcontrast2=1-spotcontrast1

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
    fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx1,spotx2]), np.array([spoty1,spoty2]), np.array([spotradius1,spotradius2]), np.array([spotcontrast1,spotcontrast2]), planetangle)
    return fitlightcurve


fit_params=[f,rp]

spotradius0=0.0
spotradius2=0.0
base=lightcurve(fit_params,period,semimajoraxis)

##
spotx1=1.0
spoty1=0.5
spotradius1=0.2
spotcontrast1=0.1

spotx2=1.0
spoty2=-0.5
spotradius2=0.2
spotcontrast2=1-spotcontrast1
fitlightcurve1=lightcurve(fit_params,period,semimajoraxis)
##

spotradius1=0.2
fitlightcurve2=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.3
fitlightcurve3=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.4
fitlightcurve4=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.5
fitlightcurve5=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.6
fitlightcurve6=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.7
fitlightcurve7=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.8
fitlightcurve8=lightcurve(fit_params,period,semimajoraxis)

spotradius1=0.9
fitlightcurve9=lightcurve(fit_params,period,semimajoraxis)


plt.figure()
plt.plot(t,base,color='black',linewidth=2.0)
plt.plot(t,fitlightcurve1,color='red')
#plt.plot(t,fitlightcurve2,color='pink')
#plt.plot(t,fitlightcurve3,color='orange')
#plt.plot(t,fitlightcurve4,color='yellow')
#plt.plot(t,fitlightcurve5,color='green')
#plt.plot(t,fitlightcurve6,color='blue')
#plt.plot(t,fitlightcurve7,color='purple')
#plt.plot(t,base,color='black',linewidth=2.0)
#plt.plot(t,fitlightcurve8,color='black')
#plt.plot(t,fitlightcurve9,color)
plt.ylim(0.969,0.975)
plt.xlim(-0.02,0.02)
plt.show()
