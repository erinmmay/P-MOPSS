import numpy as np
np.seterr('ignore')
import os

from astropy.io import fits

from setup import *


top_chip=[6,5,8,7]
bot_chip=[1,2,3,4]

def MasterFrame(path,kind,SAVEPATH,binnx,binny):
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
    print '        NUMBER OF (',kind,') FRAMES: ', n_frames
    print ' '
    full=np.empty([n_frames,4,2*ypixels/binny+ygap,xpixels/binnx])*0.0
    exp=0
    print'        -->> Reading in Calibration Frame Data...'
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            root=file.split('c')[0]
            chip=int(file.split('c')[1].split('.')[0])
            if chip in top_chip:
                c=top_chip.index(chip)
                #print '  -->>', root, chip, bot_chip[c]
                data_t=np.fliplr((fits.open(path+root+'c'+str(int(chip))+'.fits.gz')[0].data)
                                 [0:ypixels/binny,0:xpixels/binnx])
                data_b=np.flipud((fits.open(path+root+'c'+str(int(bot_chip[c]))+'.fits.gz')[0].data)
                                 [0:ypixels/binny,0:xpixels/binnx])
                full[int(exp),c,0:ypixels/binny,:]=data_t
                full[int(exp),c,ypixels/binny+ygap:,:]=data_b
            exp+=1./8.
    medfilt=np.median(full,0)
    del full
    del data_t
    del data_b
#     print'-->> Saving Calibration Frame Data...'
#     print ' '
#     variance=np.var(medfilt)
#     np.savez_compressed(SAVEPATH+kind+'.npz',medfilt=medfilt,var=variance)
    
    return medfilt

