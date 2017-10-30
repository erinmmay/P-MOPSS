from setup import *

import numpy as np
np.seterr('ignore')

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


def FindMasks(flat_path,root_flat,flat_thres,SAVEPATH,binn):
    BOXES=np.array([])
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
            p0=(paths[i])
            bbox=p0.get_extents()
            if np.abs((bbox.get_points()[0,0])-(bbox.get_points()[1,0]))> 190./binn:
                middle_of_box=(bbox.get_points()[0,0]+bbox.get_points()[1,0])/2.
                #print 'BOX #',i
                #print '   ----> MIDDLE OF BOX:', middle_of_box
                #print '   ----> WIDTH actual:', np.abs(bbox.get_points()[0,0]-bbox.get_points()[1,0])
                #print '   ----> LEFT OF BOX (actual, estimated)', bbox.get_points()[0,0],middle_of_box-width/2.
                #print '   ----> RIGHT OF BOX (actual, estimated)', bbox.get_points()[1,0],middle_of_box+width/2.
                
                #ax.add_patch(patches.PathPatch(p0, facecolor='none', ec='yellow', linewidth=2, zorder=50))
                #plt.show(block=False)
                #bbox.get_points[0,0]=MIN_x, [0,1]=MIN_y, [1,0]=MAX_x, [1,1]=MAX_y
                x0,y0,x1,y1=middle_of_box-width/(2.*binn),bbox.get_points()[0,1],middle_of_box+width/(2.*binn),bbox.get_points()[1,1]
                if top_chip[c]==6:
                    x0=x0
                    x1=x1
                if top_chip[c]==5:
                    x0=x0+(xpixels+xgap)
                    x1=x1+(xpixels+xgap)
                if top_chip[c]==8:
                    x0=x0+2.*(xpixels+xgap)
                    x1=x1+2.*(xpixels+xgap)
                if top_chip[c]==7:
                    x0=x0+3.*(xpixels+xgap)
                    x1=x1+3.*(xpixels+xgap)
                BOXES_item=Bbox(np.array([[x0,y0],[x1,y1]]))
                BOXES=np.append(BOXES,BOXES_item)
                for y in range(0,2*ypixels+ygap):
                    if y>bbox.get_points()[0,1] and y<bbox.get_points()[1,1]:
                        for x in range(0,xpixels):
                            #if x>bbox.get_points()[0,0] and x<bbox.get_points()[1,0]:
                            if x>middle_of_box-width/(2.*binn) and x<middle_of_box+width/(2.*binn):
                                masks[c,y,x]=1.0
        ax[1].imshow(masks[c,:,:], cmap=plt.cm.Greys_r, interpolation='none')
        ax[1].set_title('Generated Masks')
        plt.show(block=False)
    for x in range(ypixels,ypixels+ygap):
        masks[:,y,:]=np.nan
    #print BOXES
    BOXES=np.reshape(BOXES,(len(BOXES)/4,4))
    #print mask_edges
    np.savez_compressed(SAVEPATH+'Masks.npz',Masks=masks,paths=paths,boxes=BOXES)
    return masks

def CombineMasks(mask_full,SAVEPATH,binn):
#    import matplotlib.patches as patches
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
    paths=np.load(SAVEPATH+'Masks.npz')['boxes']#cs.collections[0].get_paths()
    for i in range(0,len(paths)):
        p0=paths[i]
        #print p0
        bbox=Bbox(np.array([[p0[0],p0[1]],[p0[2],p0[3]]]))
        #print bbox
        #print bbox.get_points
        if np.abs((bbox.get_points()[0,0])-(bbox.get_points()[1,0]))> 190./binn:
            ax0.add_patch(patches.Rectangle((p0[0],p0[1]),p0[2]-p0[0],p0[3]-p0[1], facecolor='none', ec='green', linewidth=2, zorder=50))
    ax0.set_title('Un-Combined Masks, Full Frame')
    plt.show(block=False)
    
    ##merging masks from split chips
    boxes=np.array([])
    skip_arr=np.array([])
    for i in range(0,len(paths)):
        if i in skip_arr:
            continue
        #print '----->', len(boxes)/4.
        p0=paths[i]
        bbbox=Bbox(np.array([[p0[0],p0[1]],[p0[2],p0[3]]]))
        #bbbox=p0.get_extents()
        #print i, bbox
        x0,y0,x1,y1=p0[0],p0[1],p0[2],p0[3]
        #test_point=x0+100
        #if np.abs(y1-ypixels)<20:
        #    test_point=[x0+100,y1+2*ygap]
            #print test_point
        for j in range(0,len(paths)):
            if j==i and j<len(paths)-1:
                j+=1
            if j==i and j==len(paths):
                continue
            p1=paths[j]
            x01,y01,x11,y11=p1[0],p1[1],p1[2],p1[3]
                #if p1.contains_point(test_point):
                #if test_point[0]>x01 and test_point[0]<x11:
            if (x0>x01 and x0<x11) or (x1>x01 and x1<x11):
                if np.abs(y1-y01)<2*ygap:
                #print i,j
                    skip_arr=np.append(skip_arr,j)
                    #bbox1=p1.get_extents()
                    #x01,y01,x11,y11=bbox1.get_points()[0,0],bbox1.get_points()[0,1],bbox1.get_points()[1,0],bbox1.get_points()[1,1]
                    x0n=np.nanmin([x0,x01])
                    #print 'Bottom',y01,y11, 'TOP',y0,y1
                    #print x0,x01,x0n
                    y0n=y0
                    x1n=np.nanmax([x1,x11])
                    #print x1,x11,x1n
                    #print '----'
                    y1n=y11
                    bbbox=Bbox(np.array([[x0n,y0n],[x1n,y1n]]))
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
        ax1.annotate(i,xy=(x0+100/binn,y1-500/binn),ha='center',va='center',fontsize=8,color='red',zorder=51)
    ax1.set_title('Combined Masks, Full Frame')
    plt.show(block=False)
    np.savez_compressed(SAVEPATH+'CombinedMasks.npz',mask_edges=mask_edges,boxes=boxes)
    return(mask_edges)

       
  

