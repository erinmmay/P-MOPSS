import numpy as np

from scipy.signal import medfilt

def outlierr(data,ks,sd):
    median=medfilt(data,kernel_size=ks)
    med_rm=data-median
    
    stddev=np.nanstd(med_rm)
    med_va=np.nanmedian(med_rm)
    for i in range(0,len(data)):
        if med_rm[i]>sd*stddev+med_va or med_rm[i]<med_va-sd*stddev:
            data[i]=median[i]
    
    return data

def outlierr_c(data,ks,sd):
    median=medfilt(data,kernel_size=ks)
    med_rm=data-median
    
    stddev=np.nanstd(med_rm)
    med_va=np.nanmedian(med_rm)
    
    counter=0
    for i in range(0,len(data)):
        if med_rm[i]>sd*stddev+med_va or med_rm[i]<med_va-sd*stddev:
            data[i]=median[i]
            counter+=1
    
    return counter,data

def outlierr_model(data,model,ks,sd):
    median=np.copy(model)
    med_rm=data-median
    
    stddev=np.nanstd(med_rm)
    med_va=np.nanmedian(med_rm)
    
    counter=0
    for i in range(0,len(data)):
        if med_rm[i]>sd*stddev+med_va or med_rm[i]<med_va-sd*stddev:
            data[i]=median[i]
            counter+=1
    
    return counter,data