import numpy as np
import pdb
import pandas as pd
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






def read_csv_to_correct_hlts(csv_path='Luminosity/GoldenJson/A_R_Golden.csv'):
    import tools
    path_lumi = tools.analysis_path(csv_path)
    df_lumi   =pd.read_csv(path_lumi, index_col=0) 
    df_lumi = df_lumi[['A', 'B', 'C', 'D', 'sum', 'frac', 'cumulative', 'start', 'end']]
    return df_lumi


def create_df_corrected_hlts(df, df_lumi, seed=42, verbose=True):
    """Generate uniform random numbers to make a partition of the data. The parition is made with the information of the df_lumi table.
    If an event has the random number <=end and  >start, then we assing the label of the corresponding HLT.
    If either of the muons of a candidate fired the HLT of label asigned  previously we kept it, else we discard it.

    `df` must contain columns with information is muon1 or muon2 is triggering"""

    np.random.seed(seed)
    df['rand'] = np.random.uniform(size=len(df))


    df_accepted = pd.DataFrame() 
    if verbose:
        print('\t', '\t', 'Slice','\t','%', '\t', 'Accepted', '  %')
    for HLT, row in df_lumi.iterrows():
        if HLT in ['Mu9_IP0', 'Mu9_IP3']: continue
        if HLT == 'Total': continue
        slice_ = df.query(f'{row.start}<rand<={row.end}')
        accepted = slice_.query(f'Muon1_HLT_{HLT}==1 or Muon2_HLT_{HLT}==1')
        accepted['Slice'] = HLT
        df_accepted = df_accepted.append(accepted)
        effi = '-' if  len(slice_)==0 else round(100*len(accepted)/len(slice_), 1)
        if verbose:
            print(HLT, '\t',len(slice_), '\t',round(100*len(slice_)/len(df), 1), '\t', len(accepted), '\t', effi,)
    if verbose:
        print('Initial : ', len(df))
        print('Passed  : ', len(df_accepted))
        print('Eff.    : ', round(len(df_accepted)/len(df),4))
    
    return df_accepted