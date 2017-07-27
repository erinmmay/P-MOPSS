from setup import *

import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from matplotlib.transforms import Bbox
### TOP CHIPS=6,5,8,7
### BOT CHIPS=1,2,3,4
top_chip=[6,5,8,7]
bot_chip=[1,2,3,4]

width=200.

masks=np.empty([4,2*ypixels+ygap,xpixels])*0.0

y_arr=np.linspace(0,2*ypixels+ygap,2*ypixels+ygap)
x_arr=np.linspace(0,xpixels,xpixels)
X,Y=np.meshgrid(x_arr,y_arr)


def FindMasks(flat_path,root_flat,flat_thres):
    print ' CHIP ALIGNMENT:'
    print '---------------------------------'
    print '|   6   |   5   |   8   |   7   |'
    print '---------------------------------'
    print '|   1   |   2   |   3   |   4   |'
    print '---------------------------------'
    for c in range(0,len(top_chip)):
        print ' '
        print '------------------------------'
        print ' Working on chips', top_chip[c], '&', bot_chip[c]
        print '------------------------------'
        data_t=np.fliplr(((fits.open(flat_path+root_flat+str(int(top_chip[c]))+'.fits.gz'))[0].data)[0:ypixels,0:xpixels])
        data_b=np.flipud(((fits.open(flat_path+root_flat+str(int(bot_chip[c]))+'.fits.gz'))[0].data)[0:ypixels,0:xpixels])
        data=np.empty([2*ypixels+ygap,xpixels])*np.nan
        data[0:ypixels,:]=data_t
        data[ypixels+ygap:,:]=data_b
        print '   -->>  DATA STITCHED'

        fig,ax=plt.subplots(1,2,figsize=(6.,6.))
        #plt.title('CHIPS'+str(int(top_chip[c]))+'&'+str(int(bot_chip[c])))
        im=ax[0].imshow(np.log10(data),cmap=plt.cm.Greys_r)
        #im=ax[0].colorbar()
        cs=ax[0].contour(X,Y,data,levels=[flat_thres],color='yellow',linewidth=2.0)
        ax[0].set_title('FLATS DATA')

        paths=cs.collections[0].get_paths()
        for i in range(0,len(paths)):
            p0=(cs.collections[0].get_paths()[i])
            bbox=p0.get_extents()
            if np.abs((bbox.get_points()[0,0])-(bbox.get_points()[1,0]))> 190.:
                #ax.add_patch(patches.PathPatch(p0, facecolor='none', ec='yellow', linewidth=2, zorder=50))
                #plt.show(block=False)
                for y in range(0,2*ypixels+ygap):
                    if y>bbox.get_points()[0,1] and y<bbox.get_points()[1,1]:
                        for x in range(0,xpixels):
                            if x>bbox.get_points()[0,0] and x<bbox.get_points()[1,0]:
                                masks[c,y,x]=1.0
        ax[1].imshow(masks[c,:,:], cmap=plt.cm.Greys_r, interpolation='none')
        ax[1].set_title('Generated Masks')
        plt.show(block=False)
    for x in range(ypixels,ypixels+ygap):
        masks[:,y,:]=np.nan
    np.savez('SaveData/Masks.npz',Masks=masks,paths=paths)
    return masks

def CombineMasks(mask_full):
    import matplotlib.patches as patches
    y_arr_f=np.linspace(0,2*ypixels+ygap,2*ypixels+ygap)
    x_arr_f=np.linspace(0,4*xpixels+3*xgap,4*xpixels+3*xgap)
    X,Y=np.meshgrid(x_arr_f,y_arr_f)

    fig0,ax0=plt.subplots(1,figsize=(8,8))
    cs=ax0.contourf(X,Y,mask_full[0,:,:],cmap=plt.cm.Greys_r)
    fig0.colorbar(cs,cmap=plt.cm.Greys_r)
    cs=ax0.contour(X,Y,mask_full[0,:,:],levels=[0.99],color='red',linewidth=2.0)
    ax0.set_ylim(2*ypixels+ygap,0)
    #### chip edges....
    plt.axhline(y=ypixels,color='yellow')
    plt.axhline(y=ypixels+ygap,color='yellow')
    plt.axvline(x=xpixels,color='yellow')
    plt.axvline(x=xpixels+xgap,color='yellow')
    plt.axvline(x=2*xpixels+xgap,color='yellow')
    plt.axvline(x=2*xpixels+2*xgap,color='yellow')
    plt.axvline(x=3*xpixels+2*xgap,color='yellow')
    plt.axvline(x=3*xpixels+3*xgap,color='yellow')
    ####
    paths=cs.collections[0].get_paths()
    for i in range(0,len(paths)):
        p0=paths[i]
        bbox=p0.get_extents()
        if np.abs((bbox.get_points()[0,0])-(bbox.get_points()[1,0]))> 190.:
            ax0.add_patch(patches.PathPatch(p0, facecolor='none', ec='green', linewidth=2, zorder=50))
    ax0.set_title('Un-Combined Masks, Full Frame')
    plt.show(block=False)
    
    ##merging masks from split chips
    boxes=np.array([])
    skip_arr=np.array([])
    for i in range(0,len(paths)):
        if i in skip_arr:
            continue
        p0=paths[i]
        bbbox=p0.get_extents()
        #print i, bbox
        x0,y0,x1,y1=bbbox.get_points()[0,0],bbbox.get_points()[0,1],bbbox.get_points()[1,0],bbbox.get_points()[1,1]
        if np.abs(y1-ypixels)<20:
            test_point=[x0+100,y1+2*ygap]
            #print test_point
            for j in range(0,len(paths)):
                if j==i:
                    j+=1
                p1=paths[j]
                if p1.contains_point(test_point):
                    #print i,j
                    skip_arr=np.append(skip_arr,j)
                    bbox1=p1.get_extents()
                    x01,y01,x11,y11=bbox1.get_points()[0,0],bbox1.get_points()[0,1],bbox1.get_points()[1,0],bbox1.get_points()[1,1]
                    x0n=np.nanmin([x0,x01])
                    y0n=y0
                    x1n=np.nanmax([x1,x11])
                    y1n=y11
                    bbbox=Bbox(np.array([[x0n,y0n],[x1,y1n]]))
                #elif not p1.contains_point(test_point):
                #    bbox_new=bbbox
        #else:
        #    bbox_new=bbbox
        boxes=np.append(boxes,bbbox)
    mask_edges=np.reshape(boxes,(len(boxes)/4,4))#np.empty([4,len(boxes/4.)])
    #p=0
    #for b in range(0,len(boxes)):
    #    x=b%4
    #   mask_edges[x,int(p)]=boxes[b]
    #    p+=(1./4.) 
    
    ## plotting newly merged boxes
    fig1,ax1=plt.subplots(1,figsize=(8,8))
    cs=ax1.contourf(X,Y,mask_full[0,:,:],cmap=plt.cm.Greys_r)
    fig1.colorbar(cs,cmap=plt.cm.Greys_r)
    cs=ax1.contour(X,Y,mask_full[0,:,:],levels=[0.99],color='red',linewidth=2.0)
    ax1.set_ylim(2*ypixels+ygap,0)
    #### chip edges....
    plt.axhline(y=ypixels,color='yellow')
    plt.axhline(y=ypixels+ygap,color='yellow')
    plt.axvline(x=xpixels,color='yellow')
    plt.axvline(x=xpixels+xgap,color='yellow')
    plt.axvline(x=2*xpixels+xgap,color='yellow')
    plt.axvline(x=2*xpixels+2*xgap,color='yellow')
    plt.axvline(x=3*xpixels+2*xgap,color='yellow')
    plt.axvline(x=3*xpixels+3*xgap,color='yellow')
    ####
    #paths=cs.collections[0].get_paths()
    for i in range(0,len(boxes)/4):
        x0,y0,x1,y1=mask_edges[i,0],mask_edges[i,1],mask_edges[i,2],mask_edges[i,3]
        ax1.add_patch(patches.Rectangle((x0,y0),np.abs(x0-x1),np.abs(y0-y1), facecolor='none', ec='cyan', linewidth=2, zorder=50))
        ax1.annotate(i,xy=(x0+100,y1-500),ha='center',va='center',fontsize=8,color='red',zorder=51)
    ax1.set_title('Combined Masks, Full Frame')
    plt.show(block=False)
    np.savez('SaveData/CombinedMasks.npz',mask_edges=mask_edges,boxes=boxes)
    return(mask_edges)

       
  

