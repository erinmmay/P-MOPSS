## combine double chips frames to full 8 chip frames ##
import numpy as np
np.seterr('ignore')

from setup import *

def FullFrame(n_frames,d_frames):
    FULL=np.empty([n_frames,2*ypixels+ygap,4*xpixels+3*xgap])*0.0
    if n_frames==1:
        FULL[0,:,0:xpixels]=d_frames[0,:,:]
        FULL[0,:,xpixels+xgap:2*xpixels+xgap]=d_frames[1,:,:]
        FULL[0,:,2*xpixels+2*xgap:3*xpixels+2*xgap]=d_frames[2,:,:]
        FULL[0,:,3*xpixels+3*xgap:]=d_frames[3,:,:]
    return FULL
