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