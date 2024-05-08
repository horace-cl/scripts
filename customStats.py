import scipy
import math
from scipy import stats
import numpy as np
import pandas as pd




def covariance_to_correlation(cov_matrix):
    # Calculate the standard deviations for each variable
    std_deviations = np.sqrt(np.diag(cov_matrix))
    
    # Calculate the correlation matrix
    correlation_matrix = cov_matrix / np.outer(std_deviations, std_deviations)
    
    return correlation_matrix


def correlation_to_covariance(correlation_matrix, std_deviations):
    # Calculate the covariance matrix
    cov_matrix = correlation_matrix * np.outer(std_deviations, std_deviations)
    
    return cov_matrix

def scale_covariance(cov_matrix, sigma=1):
    corr  = covariance_to_correlation(cov_matrix)
    stds  = np.sqrt(np.diag(cov_matrix))
    return correlation_to_covariance(corr, stds*sigma)

def clopper_pearson(x, n, alpha=0.32, return_errors=True):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = stats.beta.ppf
    #if isinstance(x, np.ndarray):
    #    alpha = alpha*np.ones_like(x)
    ratio = x/n
    ratio = np.nan_to_num(ratio, nan=0)
    
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    if isinstance(x, np.ndarray):
        lo = np.nan_to_num(lo, nan=0)
        hi = np.nan_to_num(hi, nan=1)
        if return_errors:
            lo = ratio - lo
            hi = hi -ratio
            hi = np.where((ratio==0) & (hi>0.5), 0, hi )
        to_return = np.array([lo, hi])
    else:
        to_return = [0.0 if math.isnan(lo) else lo, 
                    1.0 if math.isnan(hi) else hi]
        if return_errors:
            to_return[0] = ratio - to_return[0]
            to_return[1] = to_return[1] -ratio
    return to_return


def chi2_histogram(data1, data2, weights1=None, weights2=None, 
                   ensure_positive_counts=True, return_histos=False, 
                   min_prob=0,
                   ignore_bins_wzero_counts=True, **kwargs):
    """Evaluate the chi2 2 sample test by binning the histograms in the same way
    If data is weighted the uncertainty is taken as sqrt(sum(w**2))
    If ensure_positive_counts, reduce the number of bins by 1 if there is a bin with negative counts.
    Dof is the number of non-empty bins (in both histos) -1 """
    #Get initial number of weighted counts and corresponding uncertainty
    h1 = histogram_weighted(data1, weights=weights1, density=True, **kwargs)
    h2 = histogram_weighted(data2, weights=weights2, density=True, bins=h1[1])
    bin_size = np.mean(h1[1][1:]-h1[1][:-1])
    rng = [h1[1][0], h1[1][-1]]
    nbins = len(h1[0])
    
    # If number of counts is negative, reduce the number of bins by 1 
    # repeat until only positive counts
    while ( (h1[0]<0).any() or (h2[0]<0).any() ) and ensure_positive_counts:
        nbins-=1
        h1 = histogram_weighted(data1, weights=weights1, density=True, range=rng, bins=nbins)
        h2 = histogram_weighted(data2, weights=weights2, density=True, bins=h1[1] )
    
    #Evaluate numerator and denominator, and the chi2 per bin
    difference = h1[0]-h2[0]
    error_no_corr = np.hypot(h1[2], h2[2])
    if ignore_bins_wzero_counts:
        mask = np.bitwise_and(h1[0]>min_prob, h2[0]>min_prob)
        difference=difference[mask]
        error_no_corr=error_no_corr[mask]
        
    ratio = difference/error_no_corr
    chi2_list = np.power(ratio,2)
    
    #Remove nans (should occur only when there are 0 counts)
    #chi2_list = chi2_list[~np.isnan(chi2_list)]
    
    chi2 = chi2_list.sum()
    dofs = len(chi2_list)-1
    p_val = 1 - stats.chi2.cdf(chi2, dofs)
    
    
    if return_histos:
        return chi2, dofs, p_val, h1, h2

    return chi2, dofs, p_val

    
        
        
def mask_inBin(data, bin_edges, index):
    return (data>= bin_edges[index])  & (data< bin_edges[index+1])
    

    
def histogram_weighted(data, bins, weights=None,density=False,**kwargs):
    
    supported_types = [np.ndarray, pd.Series]    
    if not type(weights) in supported_types:
        if weights==None:
            weights = np.ones_like(data)
            
    
    counts, bin_edges = np.histogram(data, bins=bins, **kwargs)
    bin_size = bin_edges[1]-bin_edges[0]
    bins = len(counts)
    
    counts_weighted = np.zeros_like(counts, dtype=float)
    errors_weighted = np.zeros_like(counts, dtype=float)
    
    for i in range(bins):
        events_in = mask_inBin(data, bin_edges, i)
        counts_weighted[i] = np.sum(weights[events_in])
        errors_weighted[i] = np.sqrt(np.sum(np.power(weights[events_in], 2)))
    
    if density:
        sum_w            = np.sum(counts_weighted)
        counts_weighted /= (sum_w*bin_size)
        errors_weighted /= (sum_w*bin_size)
    
    return (counts_weighted, bin_edges, errors_weighted)