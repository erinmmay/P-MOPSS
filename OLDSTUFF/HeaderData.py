from setup import *
#start=datetime.now()
#print nchips
import numpy as np
np.seterr('ignore')

from astropy.io import fits
import os

def ReadHeader(data_path,SAVEPATH): #build arrays to store information from headers, data cubes
    file_cnt=0
    for file in os.listdir(data_path):
        if file.endswith('.fits.gz'):
            file_cnt+=1
    n_exp=file_cnt/nchips
    obs_times=np.array([])                 #holds observing dates/times (ONLY SAVED ONCE)
    exp_times=np.empty([n_exp])
    elc_noise=np.empty([nchips])                           #holds electron noise
    airmass=np.empty([n_exp])
    RA=np.empty([n_exp])
    DEC=np.empty([n_exp])
    Angle_o=np.empty([n_exp])
    Angle_e=np.empty([n_exp])
    ccd_temp=np.empty([nchips,n_exp])
    stc_temp=np.empty([nchips,n_exp])
    ion_pump=np.empty([nchips,n_exp])   
    exp_cnt=0
    for file in os.listdir(data_path):
        if file.endswith('.fits.gz'):
            head=fits.open(data_path+file)[0].header
            #head=data[0].header
            c=head['CHIP']
            if exp_cnt%20==0:
                print np.int((100*exp_cnt)/n_exp),'%'
            if c==1:
                obs_times=np.append(obs_times,str(head['DATE-OBS']+'T'+head['TIME-OBS']))
                exp_times[int(exp_cnt)]=head['EXPTIME']
                #print obs_times[int(exp_cnt)]
                airmass[int(exp_cnt)]=head['AIRMASS']
                RA[int(exp_cnt)]=head['RA-D']
                DEC[int(exp_cnt)]=head['DEC-D']
                Angle_o[int(exp_cnt)]=head['ROTANGLE']
                Angle_e[int(exp_cnt)]=head['ROTATORE']
            elc_noise[c-1]=head['ENOISE']
            ccd_temp[c-1,int(exp_cnt)]=head['TEMPCCD'+str(int(c))]
            stc_temp[c-1,int(exp_cnt)]=head['TEMPSTR']
            ion_pump[c-1,int(exp_cnt)]=head['IONPUMP']
            exp_cnt+=(1./8.)
            #head.close()
    print '-->> Header Data Read'
    print '------------------------------------------'
    print '   Observing Times: obs_times[n_exp]'
    print '   Exposure Times:  exp_times[n_exp]'
    print '   Object Airmass:  airmass[n_exp]'
    print '   RA:              RA[n_exp]'
    print '   DEC:             DEC[n_exp]'
    print '   Offset Angle:    Angle_o[n_exp]'
    print '   Encoder Angle:   Angle_e[n_exp]'
    print '   Read Noise:      elc_noise[chip]'
    print '   CCD Temperature: ccd_temp[chip,n_exp]'
    print '   Structure Temp:  stc_temp[chip_n_exp]'
    print '   Ion Pump Press:  ion_pump[chip_n_exp]'
    print '------------------------------------------'
    np.savez_compressed(SAVEPATH+'HeaderData.npz',n_exp=n_exp,obs_times=obs_times,exp_times=exp_times,airmass=airmass,RA=RA,DEC=DEC,Angle_o=Angle_o,Angle_e=Angle_e,elc_noise=elc_noise,ccd_temp=ccd_temp,stc_temp=stc_temp,ion_pump=ion_pump)
    del obs_times
    del airmass
    del elc_noise
    del ccd_temp
    del stc_temp
    del ion_pump
    #print 'TIME TO RUN: ', end-start
