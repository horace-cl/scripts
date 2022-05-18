import numpy as np
import pdb
import warnings

def create_percentile_bins(values, nbins=10):
    bin_edges = [np.percentile(values, 0)]
    for i in range(nbins):
        be = np.percentile(values, (i+1)*100/nbins)
        bin_edges.append(be)
    return bin_edges


def prior_uniform(up): 
    return np.random.uniform(high=up, size=1, )[0]


def create_accepted(data,
                    objective,
                    variables,
                    nbins=10, 
                    percentile_bins=False, 
                    density=False, 
                    w_data='None',  
                    w_objective='None', 
                    random_numbers=1, 
                    factor=1, 
                    bins_opts = dict()):
    """
    Given an initial dataframe `data` create an accep-rejected dataframe that resembles the
    shape of the `objective` dataframe using n-dimensional histograms.
    The `objective` dataframe can be the same as the `data` if any of the objective or the original df should be weigthed.
    Variables to be considered for the accept-reject must be indicated by the `variables` list.
    If density==True histograms
    """
    
    # You can pass the number of bins for each variable, 
    # therefore is must be a number or a list the same size of data columns
    if type(nbins)==int:
        nbins = [nbins for v in variables]
    
    
    # If percenile bins, the bins used are obtained independently 
    # 
    if percentile_bins:
        if density: warnings.warn('Denisty and non uniform bins could produce unexpected behaviour')
        binedges = [create_percentile_bins(data[var], nbins=nbins[i]) for i, var in enumerate(variables)]
    
    else:
        if bins_opts:
            #TODO
            raise NotImplementedError('Please implement this stuff')
        else:
            binedges = [np.linspace(
                                np.min(data[var]),
                                np.max(data[var]),
                                nbins[i]+1) 
                        for i, var in enumerate(variables)]
        
        
    if np.all(w_data=='None'): w_data=np.ones(len(data))
    if np.all(w_objective=='None'): w_objective=np.ones(len(objective))



    histo, bins    = np.histogramdd([data[var] for var in variables], 
                              weights = np.array(w_data),
                              #range = [options[var]['range'] for var in vars_],
                              #bins = [int(options[var]['bins']) for var in vars_],
                              bins = binedges,
                              density=density,
                             )

    histo_sW, bins    = np.histogramdd([objective[var] for var in variables], 
                                  weights=np.array(w_objective)*factor,
                                  #range = [options[var]['range'] for var in vars_],
                                  #bins = [int(options[var]['bins']) for var in vars_],
                                  bins = binedges,
                                 density=density,
                              )

    
    
    if density:
        ratio = histo/histo_sW
        ratio = ratio.flatten()
        ratio = ratio[~(np.isnan(ratio) | np.isinf(ratio))]
        if any(ratio<1):
            print(np.min(ratio))
            histo *= 1/np.min(ratio)
            
        
    digits = list()
    for i, var in enumerate(variables):
        digits_ = np.digitize(data[var], bins[i], right=True, )-1
        digits.append( np.where(digits_>=len(bins[i])-1, -1, digits_) )
    digits = tuple(digits)

    
    upper_random = histo[digits]
    random_numbers_l = [np.array(list(map(prior_uniform, upper_random))) for i in range(random_numbers)]
    
    cut_values = histo_sW[digits]
    accepted = [random_numbers_l[i]<cut_values for i in range(random_numbers)]

    mask = np.sum(accepted, axis=0)>=((random_numbers+1)/2)
    
    if np.all(w_data==1): 
        return data[mask]
    
    return data[mask], w_data[np.array(mask)]