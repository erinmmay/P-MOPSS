## combine double chips frames to full 8 chip frames ##
import numpy as np
np.seterr('ignore')

from setup import *

def FullFrame(n_frames,d_frames,binnx,binny):
    #binn=1
    FULL=np.empty([n_frames,int(2*ypixels/binny+ygap),int(4*xpixels/binnx+3*xgap)])*np.nan
    if n_frames==1:
        FULL[0,:,0:int(xpixels/binnx)]=d_frames[0,:,:]
        FULL[0,:,int(xpixels/binnx+xgap):int(2*xpixels/binnx+xgap)]=d_frames[1,:,:]
        FULL[0,:,int(2*xpixels/binnx+2*xgap):int(3*xpixels/binnx+2*xgap)]=d_frames[2,:,:]
        FULL[0,:,int(3*xpixels/binnx+3*xgap):]=d_frames[3,:,:]
    return FULL

def Bin_FullFrame(n_frames,d_frames,binnx,binny):
    FULL=np.empty([n_frames,2*ypixels/binny+ygap,4*xpixels/binnx+3*xgap])*np.nan
    for y in range(0,FULL.shape[1]):
        for x in range(0,FULL.shape[2]):
            if y<=(ypixels/binny):
                if x<=(xpixels/binnx):
                    FULL[0,y,x]=np.nanmedian(d_frames[0,y:y+2,x:x+2])
                if x>(xpixels/binnx)+xgap and x<=(xpixels/binnx)+xgap:
                    FULL[0,y,x]=np.nan
                if x>(xpixels/binnx)+xgap and x<=(2*xpixels/binnx)+xgap:
                    FULL[0,y,x]=np.nanmedian(d_frames[0,y:y+2,x:x+2])
                if x>(2*xpixels/binnx)+xgap and x<=(2*xpixels/binnx)+2*xgap:
                    FULL[0,y,x]=np.nan
                if x>(2*xpixels/binnx)+2*xgap and x<=(3*xpixels/binnx)+2*xgap:
                    FULL[0,y,x]=np.nanmedian(d_frames[0,y:y+2,x:x+2])
                if x>(3*xpixels/binnx)+2*xgap and x<=(3*xpixels/binnx)+3*xgap:
                    FULL[0,y,x]=np.nan
                if x>(3*xpixels/binnx)+3*xgap and x<=(4*xpixels/binnx)+3*xgap:
                    topx=np.nanmin([x+2,xpixels/binnx])
                    FULL[0,y,x]=np.nanmedian(d_frames[0,y:y+2,x:topx])
            if y>(ypixels/binny) and y<=(ypixels/binny)+ygap:
                FULL[0,y,x]=np.nan
            if y>(ypixels/binny)+ygap:
                topy=np.nanmin([y+2,2*ypixels/binny+ygap])
                if x<=(xpixels/binnx):
                    FULL[0,y,x]=np.nanmedian(d_frames[0,y:topy,x:x+2])
                if x>(xpixels/binnx)+xgap and x<=(xpixels/binnx)+xgap:
                    FULL[0,y,x]=np.nan
                if x>(xpixels/binnx)+xgap and x<=(2*xpixels/binnx)+xgap:
                    FULL[0,y,x]=np.nanmedian(d_frames[0,topy,x:x+2])
                if x>(2*xpixels/binnx)+xgap and x<=(2*xpixels/binnx)+2*xgap:
                    FULL[0,y,x]=np.nan
                if x>(2*xpixels/binnx)+2*xgap and x<=(3*xpixels/binnx)+2*xgap:
                    FULL[0,y,x]=np.nanmedian(d_frames[0,topy,x:x+2])
                if x>(3*xpixels/binnx)+2*xgap and x<=(3*xpixels/binnx)+3*xgap:
                    FULL[0,y,x]=np.nan
                if x>(3*xpixels/binnx)+3*xgap and x<=(4*xpixels/binnx)+3*xgap:
                    topx=np.nanmin([x+2,4*xpixels/binnx+3*xgap])
                    FULL[0,y,x]=np.nanmedian(d_frames[0,topy,x:topx])
     ############
                    
    fig1,ax1=plt.subplots(1,figsize=(8,8))
    x_arr=np.linspace(0,4*xpixels/binnx+3*xgap,4*xpixels/binnx+3*xgap)
    y_arr=np.linspace(0,2*ypixels/binny+ygap,42*ypixels/binny+ygap)
                    
    cs=ax1.contourf(X,Y,FULL[0,:,:],cmap=plt.cm.Greys_r)
    fig1.colorbar(cs,cmap=plt.cm.Greys_r)
    ax1.set_ylim(2*ypixels/binny+ygap,0)
    #### chip edges....
    plt.axhline(y=ypixels/binny,color='yellow')
    plt.axhline(y=ypixels/binny+ygap,color='yellow')
    plt.axvline(x=xpixels/binnx,color='yellow')
    plt.axvline(x=xpixels/binnx+xgap,color='yellow')
    plt.axvline(x=2*xpixels/binnx+xgap,color='yellow')
    plt.axvline(x=2*xpixels/binnx+2*xgap,color='yellow')
    plt.axvline(x=3*xpixels/binnx+2*xgap,color='yellow')
    plt.axvline(x=3*xpixels/binnx+3*xgap,color='yellow')
    
    plt.show(block=False)
    
    return FULL


      
            
        


