import numpy as np
import os

from astropy.io import fits

from setup import *


top_chip=[6,5,8,7]
bot_chip=[1,2,3,4]

def MasterFrame(path,kind):
    #print ' CHIP ALIGNMENT:'
    #print '---------------------------------'
    #print '|   6   |   5   |   8   |   7   |'
    #print '---------------------------------'
    #print '|   1   |   2   |   3   |   4   |'
    #print '---------------------------------'
 
    #First count the number of frames
    file_cnt=0
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            file_cnt+=1
    n_frames=file_cnt/nchips
    print ' '
    print 'NUMBER OF (',kind,') FRAMES: ', n_frames
    print ' '
    full=np.empty([n_frames,4,2*ypixels+ygap,xpixels])*0.0
    exp=0
    print'-->> Reading in Calibration Frame Data...'
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            root=file.split('c')[0]
            chip=int(file.split('c')[1].split('.')[0])
            if chip in top_chip:
                c=top_chip.index(chip)
                #print '  -->>', root, chip, bot_chip[c]
                data_t=np.fliplr((fits.open(path+root+'c'+str(int(chip))+'.fits.gz')[0].data)[0:ypixels,0:xpixels])
                data_b=np.flipud((fits.open(path+root+'c'+str(int(bot_chip[c]))+'.fits.gz')[0].data)[0:ypixels,0:xpixels])
                full[int(exp),c,0:ypixels,:]=data_t
                full[int(exp),c,ypixels+ygap:,:]=data_b
            exp+=1./8.
    medfilt=np.median(full,0)
    del full
    print'-->> Saving Calibration Frame Data...'
    print ' '
    np.savez('SaveData/'+kind+'.npz',medfilt=medfilt)
    
    return medfilt

