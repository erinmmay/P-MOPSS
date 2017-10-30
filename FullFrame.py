## combine double chips frames to full 8 chip frames ##
import numpy as np
np.seterr('ignore')

from setup import *

def FullFrame(n_frames,d_frames,binn):
    FULL=np.empty([n_frames,2*ypixels/binn+ygap,4*xpixels/binn+3*xgap])*0.0
    if n_frames==1:
        FULL[0,:,0:xpixels/binn]=d_frames[0,:,:]
        FULL[0,:,xpixels/binn+xgap:2*xpixels/binn+xgap]=d_frames[1,:,:]
        FULL[0,:,2*xpixels/binn+2*xgap:3*xpixels/binn+2*xgap]=d_frames[2,:,:]
        FULL[0,:,3*xpixels/binn+3*xgap:]=d_frames[3,:,:]
    return FULL
