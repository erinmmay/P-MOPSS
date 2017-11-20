import numpy as np
import spotrod

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
spotradius=0.0663
spotcontrast=0.2348


#contrast_arr=np.linspace(0.05,0.95,19)
#spotcontrast = 0.27
#fnum=14


fitlightcurve = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]), np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
fit_params=[planetx,planety,z,rp,r,f,spotx,spoty,spotradius,spotcontrast,planetangle]


X,Y=np.meshgrid(np.linspace(-1.0,1.0,10**3.),np.linspace(-1.0,1.0,10**3.))
X_p,Y_p=np.meshgrid(np.linspace(-2.5,2.5,10**3.),np.linspace(-2.5,2.5,10**3.))
circle=X**2.+Y**2.

spoty=-spoty
planety=-planety

spota=spotradius
spotb=spota*np.sqrt(1-(spotx**2.+spoty**2.))

spot=((X-spotx)/spota)**2.+((Y-spoty)/spotb)**2.
limb=quadraticlimbdarkening(r, u1, u2)

####################################################
norm=matplotlib.colors.Normalize(vmin=0.1 ,vmax=1.4)
#norm=matplotlib.colors.BoundaryNorm(boundaries=np.linspace(0.0,1.0,100),ncolors=100*len(limb))
#colors=matplotlib.cm.RdYlBu_r
#colors=matplotlib.cm.Greys
colors=matplotlib.cm.afmhot
scal_m=matplotlib.cm.ScalarMappable(cmap=colors,norm=norm)
scal_m.set_array([])
####################################################

rsq=r**2.

contrast_arr=np.empty([20])*np.nan

#bins_used=[3,4,5,6,7,8,9,10,11,12,15,16,19]
#for b in bins_used:
#    contrast_arr[b]=(np.load(path+name+'/'+str(int(b))+'_LightCurve_fits.npz')['fit_params'])[9]
#print contrast_arr


print 'NUMBER OF FRAMES: ',len(time)
for j in range(0,len(time)):
#for j in range(45,46):
#for b in bins_used:
    u1 = q0_func(bin_wav[b])
    u2 = q1_func(bin_wav[b])
    limb=quadraticlimbdarkening(r, u1, u2)
    print '          -->> Working on frame', j
    planet=(X_p-planetx[j])**2.+(Y_p-planety[j])**2.
    fig=plt.figure(1,figsize=[12,6])
    p0=plt.figure(1,figsize=[9,9])
    plt.clf()
    gs1=gridspec.GridSpec(2,6)
    gs1.update(wspace=0.0,hspace=0.0)
    p0=plt.subplot(gs1[:,0:3])
    p0=plt.contour(Y,X,circle,levels=[1.0],linewidths=1.,colors='black',zorder=0)
#    for i in range(0,len(r)):
#        p0=plt.contour(Y,X,circle,levels=[r[i]**2.],colors=scal_m.to_rgba(limb[i]))
    p0=plt.contour(Y,X,circle,levels=rsq,colors=scal_m.to_rgba(limb),zorder=1) 
#    p0=plt.contour(Y,X,spot,levels=[spotradius**2.],linewidths=0.25,colors='black',zorder=3)
#    p0=plt.contourf(Y,X,spot,levels=[0,spotradius**2],colors=scal_m.to_rgba([contrast_arr[b]]),alpha=0.75,zorder=2)
    p0=plt.contour(Y,X,spot,levels=[1.0],linewidths=0.25,colors='black',zorder=3)
    p0=plt.contourf(Y,X,spot,levels=[0,1.0],colors=scal_m.to_rgba([contrast_arr[b]]),alpha=0.75,zorder=2)
    p0=plt.contourf(Y_p,X_p,planet,levels=[0,rp**2.],colors='skyblue',zorder=4)
    p0=plt.contour(Y_p,X_p,planet,levels=[rp**2.],linewidth=1.,colors='black',zorder=5)
    p0=plt.plot(planety,planetx,'--',linewidth=0.5,color='skyblue',alpha=0.5)
    p0=plt.plot(planety,planetx+rp,'-',linewidth=0.5,color='skyblue',alpha=0.5)
    p0=plt.plot(planety,planetx-rp,'-',linewidth=0.5,color='skyblue',alpha=0.5)
    p0=plt.plot(planety,planetx+rp/2.,'-',linewidth=2,color='blue')
    p0=plt.plot(planety,planetx-rp/2.,'-',linewidth=2,color='blue')
    p0=plt.ylim(-2.5,2.5)
    p0=plt.xlim(-2.5,2.5)
    p0=plt.ylim(-1.5,1.5)
    p0=plt.xlim(-1.5,1.5)
    p0=plt.xticks([],[])
    p0=plt.yticks([],[])
    p1=plt.subplot(gs1[:,3:])
    p1=plt.plot(t[:j+1],fitlightcurve[:j+1],'k-',linewidth=2.0)
    p1=plt.plot(t[:j+1],flux[:j+1],'.',color='steelblue',markersize=6)
    p1=plt.xlabel('Time From Center of Transit [days]',fontsize=15)
    p1=plt.ylabel('Relative Flux')
    p1=plt.figtext(0.95,0.5,'Relative Flux', rotation='vertical', va='center',fontsize=15)
    p1=plt.xticks([-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15],[-0.2,' ', -0.1,' ', 0.0,' ',0.1,' '],fontsize=10)
    p1=plt.yticks(fontsize=10)
    p1=plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    p1=plt.xlim(-0.2,0.15)
    p1=plt.ylim(0.965,1.005)
    plt.savefig('/Users/ermay/Desktop/Wasp52/gif/'+str(int(j))+'.png')
    plt.close()
