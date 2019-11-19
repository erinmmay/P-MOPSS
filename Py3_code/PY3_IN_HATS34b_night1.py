import numpy as np

################
obs_date='ut20180914'                           #observation date
obj_name='Hats34'                                #object name   
midtime=['2018-09-15T03:39:00.00']                 #time of midtransit, WAsp-4b

obj_skip=[]
binnx=2
binny=2
#################### CONSTANTS - DON'T CHANGE
Mjup=1.89*10**30  #grams
Rjup=6.92*10**9  #cm
                                                                                                                            

Grav=6.67*10**-8.
Msun=1.989*10**33.
Rsun=6.957*10**10.
####################

q0=[0.9459,0.7502,0.5424,0.4330,0.3385,0.1813]#,0.0640,0.0683] #Quad Limb Dark 1
q1=[-0.0850,0.0799,0.2060,0.2443,0.2528,0.2953]#,0.3519,0.2893] #Quad Limb Dark 2  


mp=0.941   #mass of planet in jup masses
dmp=0.0720

rp=(1.43)        #radius of planet in jupiter(or jup with *Rjup/Rs)
rpdn=0.00
rpup=1.00

per=2.106   #period in days
per_e=0.01

inc=82.28  #orbital inclination
inc_e=1.5

ecc=0.0  #eccentricity
w=0.0

t0=0.0    #don't change this - used as a starting value                                                                          
t0dn=-0.1
t0up=0.1

Ms=0.955*Msun
Rs=0.980*Rsun
Ms_e=0.031*Msun
Rs_e=0.047*Rsun

Ts=5380
dTs=73

smacm=(((per*24.*60.*60.)**2.*Grav*Ms)/(4*np.pi**2.))**(1./3.)
sma=smacm/Rs
rp*=Rjup/Rs

