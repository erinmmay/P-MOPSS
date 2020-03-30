import numpy as np
from datetime import datetime
import sys
import os


p_name='Hats34'                 #PLANET NAME - needs to match your file structure
obs_date='ut20180914'          #Observation datae -- needs to match the file structure


#SAVEPATH='/Volumes/ermay_ext/Magellan_IMACS/'       #homedirectory to the data (directory above the dates)
SAVEPATH='/Users/mayem1/Desktop/Magellan/'


fullpath0=SAVEPATH+obs_date+'/SaveData_'+p_name+'/'  #directory that the LCS are in (ex. SaveData_Hats8b_ap5)
fullpath=SAVEPATH+obs_date+'/SaveData_'+p_name+'/'

CON = 0
ap0 = 3

runname='testing'+'_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'_new_q1q2fit'  #appends a string to generated directories below for organizational reasons


### do not change ###
sys.path.insert(0,fullpath) 
#from SystemCons import *
from PY3_IN_HATS34b_night1 import *
######################


nwalkers=200  #number of walkers in your MCMC chain
nburnin=1000  #number of steps in your burnin
nsteps=2500   #number of steps in production chain

c=0   #c=0 is white light, c=1 is binned, if c=1, needs to have c=0 ran for the center of transit
width=200   #width of bins running if c=1

par_n = ['t0','rp']#,'c1','c2']   #light curve parameters to fit for - options - t0,per,rp,sma,inc,ecc,w,c1,c2
in_par_c = np.array([t0,rp])#,0.0,0.0])   #note: initial limb darkening will get set in main code
in_par_d = np.array([t0dn,rpdn])#,-1.5,-1.5])   #lower bound for fit parameters
in_par_u = np.array([t0up,rpup])#,1.5,1.5])   #upper bound for fit parameters

#if running a binned set of fits, you'll want to read in the t0 from the white light curve below
if c>0:
    t0 = (np.load(fullpath0+'/emcee_run_whitelc_CON'+str(CON)+'_AP'+str(int(ap0*100)).zfill(3)+'.npz')['best_params'])[0]

#note: internal built in bounds for limb darkening are c1>c2 and c1+c2<1 

#Baseline Model options
BL_order=2  #order of polynomial baseline, typically 1,2, or 3.

#Noise model options
useFW=True  #decorrelate against FWHM/seeing
fwhm0=2.5   #average seeing/fwhm value to center on

useBG=False  #decorrelate against background
bg0=0.8   #average background value to center on

useXp=True  #decorrelate against spatial (X) centroid
useYp=True  #decorrelate against spectral (Y) centroid

x0=0.0  #average spatial centroid
y0=0.0  #average spectral centroid 

### Option to mask outliers the data
### ---- outlier detection is done on the data array and applied to noise model arrays
outlier = False
runtwce = True
ks = 7  #kernel_size for outlier detection, must be odd
sig = 3  #sigma level for outlier detection (3 or 4)


#### Below: creates save directorys that the code expects###

if not os.path.exists(fullpath+'LCFITS'):      
    os.mkdir(fullpath+'LCFITS')

os.chdir(fullpath+'LCFITS/')

mydir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S'+'_'+runname)

if not os.path.exists(fullpath+'LCFITS/'+mydir):
    os.mkdir(fullpath+'LCFITS/'+mydir)
    
os.chdir(fullpath+'LCFITS/'+mydir)

if not os.path.exists(fullpath+'LCFITS/'+mydir+'/LCfit_plots'):
    os.mkdir(fullpath+'LCFITS/'+mydir+'/LCfit_plots')

if not os.path.exists(fullpath+'LCFITS/'+mydir+'/LCfit_files'):
    os.mkdir(fullpath+'LCFITS/'+mydir+'/LCfit_files')








