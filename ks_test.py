import numpy as np
import pandas as pd
from scipy.stats import ks_2samp as ks_2samp_sci
import pdb

def create_cdf(data1, data2,  weights1=None, weights2=None):
    
    supported_types = [np.ndarray, pd.Series]
    #Here it tries to set weigths=1 when user does not pass them
    #Likely to break if input is not np.array
    if not type(weights1) in supported_types:
        if weights1==None:
            weights1 = np.ones_like(data1)
            
    if not type(weights2) in supported_types:
        if weights2==None:
            weights2 = np.ones_like(data2)
    
    x = np.unique(np.concatenate([data1, data2]))
    weights1_n = weights1 / np.sum(weights1) * 1.
    weights2_n = weights2 / np.sum(weights2) * 1.
    
    inds1 = np.searchsorted(x, data1)
    inds2 = np.searchsorted(x, data2)
    
    w1 = np.bincount(inds1, weights=weights1_n, minlength=len(x))
    w2 = np.bincount(inds2, weights=weights2_n, minlength=len(x))
    
    F1 = compute_cdf(w1)
    F2 = compute_cdf(w2)
    
    return F1, F2
    

def compute_cdf(ordered_weights):
    #https://github.com/arogozhnikov/hep_ml/blob/52e0156a39b02936e9a82569019201d7c49752b5/hep_ml/metrics_utils.py#L112
    """Computes cumulative distribution function (CDF) by ordered weights,
    be sure that sum(ordered_weights) == 1.
    Minor difference: using symmetrized version
    F(x) = 1/2 (F(x-0) + F(x+0))
    """
    return np.cumsum(ordered_weights) - 0.5 * ordered_weights



def ks_2samp_weighted(data1, data2, weights1=None, weights2=None, p_value=True):
    #import pdb
    #https://github.com/arogozhnikov/hep_ml/blob/master/hep_ml/metrics_utils.py#L224
    """
    Kolmogorov-Smirnov distance, almost the same as ks2samp from scipy.stats, but this version supports weights.
    :param data1: array-like of shape [n_samples1]
    :param data2: array-like of shape [n_samples2]
    :param weights1: None or array-like of shape [n_samples1]
    :param weights2: None or array-like of shape [n_samples2]
    :return: float, Kolmogorov-Smirnov distance.
    """
    supported_types = [np.ndarray, pd.Series]
    #Here it tries to set weigths=1 when user does not pass them
    #Likely to break if input is not np.array
    if not type(weights1) in supported_types:
        if weights1==None:
            weights1 = np.ones_like(data1)
            
    if not type(weights2) in supported_types:
        if weights2==None:
            weights2 = np.ones_like(data2)
    
    x = np.unique(np.concatenate([data1, data2]))
    weights1_n = weights1 / np.sum(weights1) * 1.
    weights2_n = weights2 / np.sum(weights2) * 1.
    
    inds1 = np.searchsorted(x, data1)
    inds2 = np.searchsorted(x, data2)
    
    #pdb.set_trace()
    
    w1 = np.bincount(inds1, weights=weights1_n.astype(np.float32), minlength=len(x))
    w2 = np.bincount(inds2, weights=weights2_n.astype(np.float32), minlength=len(x))
    
    F1 = compute_cdf(w1)
    F2 = compute_cdf(w2)
    distance = np.max(np.abs(F1 - F2))
    
    if p_value:
        #Kish effective sample size
        #https://sawtoothsoftware.com/help/lighthouse-studio/manual/effective_sample_size.html
        #https://en.wikipedia.org/wiki/Effective_sample_size#Weighted%20samples
        dat1_eq = np.sum(weights1)**2/np.sum(weights1**2)
        dat2_eq = np.sum(weights2)**2/np.sum(weights2**2)
        #dat1_eq = np.sum(weights1)
        #dat2_eq = np.sum(weights2)
        factor  = dat1_eq*dat2_eq/(dat1_eq+dat2_eq)
        #factor = len(data1)*len(data2)/(len(data1)+len(data2))
        z = distance*np.sqrt(factor)
        p_val = KolmogorovProb(z)
        return distance, p_val
    
    return distance


def KolmogorovProb(z):
    #https://root.cern/doc/master/TMath_8cxx_source.html#l00656
    fj = [-2,-8,-18,-32]
    w  = 2.50662827
    c1 = -1.2337005501361697
    c2 = -11.103304951225528
    c3 = -30.842513753404244
    u  = np.abs(z)
    
    if u<0.2:
        p = 1
    elif u<0.755:
        v = 1./(u*u);
        p = 1 - w*(np.exp(c1*v) + np.exp(c2*v) + np.exp(c3*v))/u
    elif u<6.8116:
        r = [0, 0,0,0]
        v = u*u
        maxj = np.max([1,int(round(3/u))]);
        
        for j in range(maxj):
            r[j] = np.exp(fj[j]*v)
        p = 2*(r[0] - r[1] +r[2] - r[3])
    else:
        p=0
    return p