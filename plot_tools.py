import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
plt.style.use(hep.style.CMS)

#hep.CMS.label(<text>, data=<True|False>, lumi=50, year=2017)
# Just experiment label and <text> such as 'Preliminary' or 'Simulation'
#hep.CMS.text(<text>)

import ks_test
#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    tf.config.experimental.set_memory_growth(gpus[0],True)
#import zfit
from copy import deepcopy
import pdb
import tools 
import json

def histos_opts(v=0):
    with open(tools.analysis_path(f'scripts/histograms_binning/v{v}.json'), 'r') as jj:
        return json.load(jj)
    

pretty_names = dict(
    cosThetaKMu = r'$\cos \theta_{\ell}$',
    Bpt = '$B$ pT  [GeV/c]',
    kpt = '$K$ pT  [GeV/c]',
    mu1_pt = '$\mu_{1}$ pT  [GeV/c]',
    mu2_pt = '$\mu_{2}$ pT  [GeV/c]',
    prob = '$SV_{prob}$',
    cosA = r'$\cos \alpha$',
    signLxy = r'$L_{xy}/\sigma_{Lxy}$',
    PDL = 'PDL  [cm]',
    ePDL = r'$\sigma_{PDL}$ [cm]',
    BMass = 'B mass  [GeV/$c^2$]',
    DiMuMass = '$\mu^+\mu^-$ mass  [GeV/$c^2$]',
    mu1_eta = '$\mu_{1}$ $\eta$',
    mu2_eta = '$\mu_{2}$ $\eta$',
    mu1_IP_sig = '$\mu_{1}$ IP/$\sigma_{IP}$',
    mu2_IP_sig = '$\mu_{2}$ IP/$\sigma_{IP}$',
    fit_eta = '$B^+ \eta$',
    fit_k_eta = '$K^+ \eta$',
    fit_l1_eta = r'$\mu_{1} \eta$ - (fitted)',
    fit_l2_eta = r'$\mu_{2} \eta$ - (fitted)',
)


######################################## WEIGHTED 1D HISTOGRAMS  ########################################
def mask_inBin(data, bin_edges, index):
    events_in = (data>= bin_edges[index])  & (data< bin_edges[index+1])
    return events_in

def mask_underflow(data, bin_edges):
    events_in = (data< bin_edges[0])
    return events_in

def mask_overflow(data, bin_edges):
    events_in = (data>= bin_edges[-1])
    return events_in


def hist_weighted(data, bins, weights=None, axis=None, only_pos=False, density=False, **kwargs):    

    supported_types = [np.ndarray, pd.Series]
    #Here it tries to set weigths=1 when user does not pass them
    #Likely to break if input is not np.array
    if not type(weights) in supported_types:
        if weights==None:
            weights = np.ones_like(data)
            
            
    if 'range' in kwargs:
        hist_opts = {'range':kwargs['range']}
        del kwargs['range']
    else:
        hist_opts = {}
    
    counts, bin_edges = np.histogram(data, bins=bins, **hist_opts)
    
    bins = len(counts)
    bin_mean = (bin_edges[1:]+bin_edges[:-1])/2 
    bin_size = bin_edges[1]-bin_edges[0]
    
    events_under = mask_underflow(data, bin_edges)
    events_over  = mask_overflow(data,  bin_edges)
    
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
        
    if  only_pos:
        non_zero        = counts_weighted>0
        bin_mean        = bin_mean[non_zero] 
        counts_weighted = counts_weighted[non_zero]
        errors_weighted = errors_weighted[non_zero]
    
    
    hist_type = kwargs.get('hist_type', 'error')
    if 'hist_type' in kwargs:
        del kwargs['hist_type']
    if axis:
        line_style = kwargs.get('ls', 'none')
        if line_style!= 'none': del kwargs['ls']
            
        if hist_type=='bar':
            axis.bar(bin_mean, 
                    counts_weighted, 
                    width=bin_size, 
                    **kwargs)
        elif hist_type=='step':
            axis.step(bin_mean, 
                    counts_weighted, 
                    where='mid',
                    #width=bin_size, 
                    **kwargs)
        else:
            axis.errorbar(x = bin_mean,  xerr=bin_size/2,
                      y = counts_weighted, yerr=errors_weighted,
                      ls = line_style,
                      **kwargs)
    else:
        if 'ls' in kwargs: line_style = kwargs['ls'];del kwargs['ls']
        else: line_style = 'none'
                     
                     
        if hist_type=='bar':
            plt.bar(bin_mean, 
                    counts_weighted, 
                    width=bin_size, 
                   **kwargs)
        elif hist_type=='step':
            plt.step(bin_mean, 
                    counts_weighted, 
                    where='mid',
                    #width=bin_size, 
                    **kwargs)
        else:        
            plt.errorbar(x  = bin_mean,        xerr = bin_size/2,
                     y  = counts_weighted, yerr = errors_weighted,
                     ls = line_style,
                      **kwargs)
        
    if any(events_under):
        under_cou = np.sum(weights[events_under])
        under_err = np.sqrt(np.sum(np.power(weights[events_under], 2)))
        print(f'Underflow (<{np.round(bin_edges[0],3)})')
        print('\t', round(under_cou, 2), '+-', round(under_err,2) )
        print('\tUnweighted ', len(weights[events_under]), '\n' )
        
    if any(events_over):
        over_cou = np.sum(weights[events_over])
        over_err = np.sqrt(np.sum(np.power(weights[events_over], 2)))
        print(f'Overflow  (>={np.round(bin_edges[-1],3)})')
        print('\t', round(over_cou, 2), '+-', round(over_err,2) )
        print('\tUnweighted ', len(weights[events_over]), '\n' )
    
        
    return counts_weighted, bin_edges, errors_weighted
######################################## WEIGHTED 1D HISTOGRAMS  ########################################






######################################## HISTOGRAMS WITH OVERFLOW AND UNDERFLOW INFO ########################################
def hist(data:'array like data', 
         low_percentile  = 5,
         high_percentile = 95,
         axis=None,
         hist_type:'hist, bar or error'='hist',
         **kwargs):
    
    
    min_ = np.percentile(data, low_percentile)
    max_ = np.percentile(data, high_percentile)
    
    #Trying to define in a "better" way the
    #range definition of histograms
    max_distance = np.max(data) - np.min(data)
    if max_distance<2*(max_ - min_):
        max_, min_ = np.max(data), np.min(data)
    
    #Extract arguments for binning
    hist_opts    = dict() 
    hist_opts['range'] = [min_, max_]
    posible_opts =  ['bins', 'density', 'range'] 
    for opt in posible_opts:
        if opt in kwargs: hist_opts[opt] = kwargs[opt]; del kwargs[opt]
    
    counts, bin_edges =  np.histogram(data, **hist_opts)
    
    
    under_flow = len(data[data<bin_edges[0]])
    if under_flow:
        print(f'Underflow (<{np.round(bin_edges[0],3)})  -  {under_flow} Events')
    over_flow = len(data[data>=bin_edges[-1]])
    if over_flow:
        print(f'Overflow (>={np.round(bin_edges[-1],3)})  -  {over_flow} Events')
        
    
    
    bin_mean = (bin_edges[1:]+bin_edges[:-1])/2
    bin_size = bin_edges[1] - bin_edges[0]
    
    #print(kwargs)
    if axis:
        if hist_type=='hist':
            obj_ = axis.hist(data, 
                             bins=bin_edges, 
                             density = hist_opts.get('density', False),
                             **kwargs)
        elif hist_type=='bar':
            obj_ = axis.bar(bin_mean, 
                            counts, 
                            width=bin_size, 
                            **kwargs)
        elif hist_type=='error':
            obj_ = axis.errorbar(bin_mean, counts, 
                                 xerr=bin_size/2,
                                 yerr=np.sqrt(counts),
                                 **kwargs)
    else:
        if hist_type=='hist':
            obj_ = plt.hist(data, 
                            bins=bin_edges, 
                            density = hist_opts.get('density', False),
                            **kwargs)
        elif hist_type=='bar':
            obj_ = plt.bar(bin_mean, 
                           counts, 
                           width=bin_size, 
                           **kwargs)
        elif hist_type=='error':
            y_err = np.sqrt(counts)
            if hist_opts.get('density', False):
                len_ = data[(data>=bin_edges[0]) & (data<=bin_edges[-1])]
                y_err /= (len_*bin_size)
            obj_ = plt.errorbar(bin_mean,
                                counts, 
                                xerr=bin_size/2,
                                yerr=np.sqrt(counts),
                                **kwargs)
    
        
    return counts, bin_edges, obj_
    
    
    
    
######################################## HISTOGRAMS WITH OVERFLOW AND UNDERFLOW INFO ########################################





######################################## 1D HISTOGRAMS GIVEN HEIGHT AND BINS EDGES  ########################################
def hist_from_heights(heights, bin_edges, axis=None, **kwargs):
    bin_mean = (bin_edges[1:]+bin_edges[:-1])/2
    width = bin_edges[1]-bin_edges[0]
    if axis:
        fig = axis.bar(bin_mean, heights, width=width, **kwargs)
    else:
        fig = plt.bar(bin_mean, heights, width=width, **kwargs)
    return fig
######################################## 1D HISTOGRAMS GIVEN HEIGHT AND BINS EDGES  ########################################








def prepare_histogram(data, bins, range_, errors='Poisson'):

    histogram = np.histogram(data, bins, range=range_)
    bin_mean = (histogram[1][1:]+histogram[1][:-1])/2
    bin_size = histogram[1][1]-histogram[1][0]
    n_events = np.sum(histogram[0])
    scale = bin_size*n_events

    if errors.title() == 'Poisson':
        y_err = np.sqrt(histogram[0])
    elif errors.title() == "Binomial":
        y_err = np.sqrt(histogram[0]*(1-histogram[0]/n_events))
    else:
        raise NotImplementedError("Only Poisson and Binomial errors implemented, if you want more please update:\n `../hcl/scripts/plot_tools.py`")

    return histogram, y_err, bin_mean, bin_size, scale 


def create_pulls(histogram, pdf, y_err, integrate=True):
    print('Creating Pulls')
    expected_events_m = np.zeros_like(histogram[0], dtype=np.float128)
    for i in range(len(histogram[0])):
        if int(100*(1+i)/len(histogram[0]))%10==0:
            print('\t', i+1, f'/ {len(histogram[0])} ')
        #pdb.set_trace()
        if integrate:
            val = pdf.integrate(limits=[histogram[1][i],histogram[1][i+1]]).numpy()[0]
        else:
            val = pdf.pdf((-histogram[1][i]+histogram[1][i+1])/2).numpy()
        expected_events_m[i] = val*np.sum(histogram[0])
    pull = (histogram[0]-expected_events_m)/y_err
    zfit.util.cache.clear_graph_cache()
    return pull


def get_parameter(pdf, sub_string, sub_string2=None):
    """Get a parameter of the `pdf` whose name contains the `sub_string` (no special characters)"""
    sub_string = sub_string.lower()
    names=list()
    index=list()
    for i,p in enumerate(pdf.get_params()):
        if sub_string in p.name.lower().replace('_', ''):
            if sub_string2 and not sub_string2 not in p.name.lower().replace('_', ''): continue
            names.append(p.name)
            index.append(i)
    if len(names) == 0:
        raise NotFoundError(f'No parameter contains `{sub_string}`')
    elif len(names) > 1:
        print('WARNING!\n', f'More than one match, using the first one')
        print('MATCHES:   ', names)
    return pdf.get_params()[index[0]]





"""def create_grid_simple(fig, nrows, ncols):
    y_size = 100*nrows
    x_size = ncols
    axes = []
    for row in range(nrows):
        rows_ = []
        for col in range(ncols):
            ax  = plt.subplot2grid(shape = (y_size,x_size), loc = (100*row,col),
                                   rowspan = 70, fig = fig)
            axp = plt.subplot2grid(shape = (y_size,x_size), loc = (100*row+72,col),
                                   rowspan = 18, fig = fig)
            rows_.append((ax, axp))
        axes.append(rows_)
        
    return axes"""





def create_axes_for_pulls(fig, split = 70, space_between = 2):
    ax  = plt.subplot2grid(shape = (100,1), loc = (0,0),
                           rowspan = split, fig = fig)
    axp = plt.subplot2grid(shape = (100,1), loc = (split+space_between,0),
                           rowspan = 100-(split+space_between), fig = fig)
    
    axp.get_shared_x_axes().join(axp, ax)
    
    ax.set_xticklabels([])
    return ax, axp



def create_grid_for_pulls(fig, nrows, ncols, 
    space_between=2, 
    rowspan1=70, 
    rowspan2=18):

    y_size = 100*nrows
    x_size = ncols
    axes = []
    for row in range(nrows):
        rows_ = []
        for col in range(ncols):
            ax  = plt.subplot2grid(shape = (y_size,x_size), loc = (100*row,col),
                                   rowspan = rowspan1, fig = fig)
            axp = plt.subplot2grid(shape = (y_size,x_size), loc = (100*row+rowspan1+space_between,col),
                                   rowspan = rowspan2, fig = fig)
            axp.get_shared_x_axes().join(axp, ax)
            ax.set_xticklabels([])
            rows_.append((ax, axp))
        axes.append(rows_)
        
    return axes


def create_grid_for_pulls_Spec(fig, nrows, ncols):
    y_size = 100*nrows
    x_size = ncols
    axes = []
    for row in range(nrows):
        rows_ = []
        for col in range(ncols):
            ax  = plt.subplot2grid(shape = (y_size,x_size), loc = (100*row,col),
                                   rowspan = 70, fig = fig)
            axp = plt.subplot2grid(shape = (y_size,x_size), loc = (100*row+72,col),
                                   rowspan = 18, fig = fig)
            axp.get_shared_x_axes().join(axp, ax)
            ax.set_xticklabels([])
            rows_.append((ax, axp))
        axes.append(rows_)
        
    return axes







def textParams2(minimum, ncol=2, clean=True):
    texts = ['' for c in range(ncol)]
    n_params = len(minimum.hesse())
    
    for indx_p, (param, error) in enumerate(minimum.hesse().items()):
        
        col = indx_p%ncol
        text=''
        param_value = minimum.params[param]['value']
        
        err = error['error']
        r=1
        
        if clean or (param.name.startswith('$') and param.name.endswith("$")):
            name = param.name
        else:
            name = param.name.split('_')[:-1]
            name = '$'+'_'.join(name)+'$'
            if name[0]!='$': name = '$'+name

        if err<1:
            r += int(np.round(-np.log10(np.abs(err))))

        if 'Y' in name:
            

            text+= name.split('Bin')[0]
            if param.value().numpy()>=100000:
                text+= ' = '+ "{:.2e}".format(param_value) + ' $\pm$ '
                text+=  "{:.2e}".format(error["error"]) +'\n'
            else:
                text+= ' = '+str(int(param_value)) + '$\pm$'
                text+= f' {int(error["error"])}\n'

        else:
            text+= name.split('Bin')[0]
            text+= f' = {round(param_value,r)} ' + '$\pm$'
            text+= f' {round(error["error"], r)}\n'

        texts[col]+=text
        
    texts = [t.strip() for t in texts]
        
    return texts


def textParams(minimum, ncol=2, clean=True):
    texts = ['' for c in range(ncol)]
    n_params = len(minimum.params)
    
    for indx_p, (param, result_) in enumerate(minimum.params.items()):
        
        col = indx_p%ncol
        text=''
        param_value = minimum.params[param]['value']
        
        if 'minuit_hesse' in result_: err = result_['minuit_hesse']['error']
        elif 'hesse_np' in result_: err = result_['hesse_np']['error']
        else: err = -1
        r=1
        
        if clean or (param.name.startswith('$') and param.name.endswith("$")):
            name = param.name
        else:
            name = param.name.split('_')[:-1]
            name = '$'+'_'.join(name)+'$'
            if name[0]!='$': name = '$'+name

        if err<0:
            r += 1+int(np.round(-np.log10(np.abs(param_value))))

        elif err<1:
            r += int(np.round(-np.log10(np.abs(err))))


        if 'Y' in name:
            

            text+= name.split('Bin')[0]
            if param.value().numpy()>=100000:
                text+= ' = '+ "{:.2e}".format(param_value) 
                if err > 0: text+=  ' $\pm$ ' + "{:.2e}".format(err)
                text+='\n'

            else:
                text+= ' = '+str(int(param_value))
                if err > 0: text+=  '$\pm$' + f' {int(err)}'
                text+='\n'

        else:
            text+= name.split('Bin')[0]
            text+= f' = {round(param_value,r)} ' 
            if err>0 : text+= '$\pm$'+ f' {round(err, r)}'
            text+='\n'

        texts[col]+=text
        
    texts = [t.strip() for t in texts]
        
    return texts




def textParams_from_fixed_model(model, ncol=1, clean=True):
    texts = ['' for c in range(ncol)]
    for indx_p, (name_p, param) in enumerate(model.params.items()):
        
        col = indx_p%ncol
        text=''
        param_value = param.value().numpy()
        
        r=2
        
        if clean or (name_p.startswith('$') and name_p.endswith("$")):
            name = name_p
        else:
            name = name_p.split('_')[:-1]
            name = '$'+'_'.join(name)+'$'
            if name[0]!='$': name = '$'+name


        if 'Y' in name:    
            text+= name.split('Bin')[0]
            if param.value().numpy()>=100000:
                text+= ' = '+ "{:.2e}".format(param_value) 
                text+='\n'
            else:
                text+= ' = '+str(int(param_value))
                text+='\n'

        else:
            #if param_value!=0:  r = int(np.ceil(np.abs(np.log10(param_value))))
            text+= name.split('Bin')[0]
            text+= f' = {round(param_value,r)} ' 
            text+='\n'

        texts[col]+=text


        
    texts = [t.strip() for t in texts]
        
    return texts




def textParams_from_model(model, ncol=1, clean=True):
    texts = ['' for c in range(ncol)]
    n_params = len(model.get_params())
    
    if n_params==0:
        return textParams_from_fixed_model(model, ncol, clean)

    for indx_p, param in enumerate(model.get_params()):
        
        col = indx_p%ncol
        text=''
        param_value = param.value().numpy()
        
        r=2
        
        if clean or (param.name.startswith('$') and param.name.endswith("$")):
            name = param.name
        else:
            name = param.name.split('_')[:-1]
            name = '$'+'_'.join(name)+'$'
            if name[0]!='$': name = '$'+name


        if 'Y' in name:
            
            text+= name.split('Bin')[0]
            if param.value().numpy()>=100000:
                text+= ' = '+ "{:.2e}".format(param_value) 
                text+='\n'

            else:
                text+= ' = '+str(int(param_value))
                text+='\n'

        else:
            #if param_value!=0:  r = int(np.ceil(np.abs(np.log10(param_value))))
            text+= name.split('Bin')[0]
            text+= f' = {round(param_value,r)} ' 
            text+='\n'

        texts[col]+=text


        
    texts = [t.strip() for t in texts]
        
    return texts








def plot_pull(h, pdf, xlabel, axis, return_chi2=False, integrate=False, return_expected_evts=False):

    try:
        pdf.norm_range.spaces
        limits = [pdf.norm_range.spaces[0].lower.flatten()[0],
                  pdf.norm_range.spaces[-1].upper.flatten()[0]
        ]
    except AttributeError:
        limits = pdf.norm_range.limit1d

        
    bin_mean = (h[1][1:]+h[1][:-1])/2
    bin_sz = h[1][1]-h[1][0]
    n_events = np.sum(h[0])
    y_err = np.sqrt(h[0]*(1-h[0]/n_events))
    y_err = np.sqrt(h[0])
    expected_events = bin_model(pdf, h[1], integrate=integrate)*n_events
        
    #pull = (h[0]-expected_events)/np.sqrt(expected_events)
    pull = (h[0]-expected_events)/np.sqrt(h[0])
    axis.errorbar(bin_mean, pull, yerr=1, ls='none', capsize=1, color='black')
    axis.scatter(bin_mean, pull, color='black', s=40)
    axis.plot(limits, [3,3], ls='--', color='grey')
    axis.plot(limits, [-3,-3], ls='--', color='grey')
    axis.plot(limits, [0,0], ls=':', color='grey')
    pull_max=np.max(np.abs(pull[np.isfinite(pull)]))
    if pull_max>3:
        axis.set_ylim((-pull_max*1.2, pull_max*1.2))
        axis.set_yticks(np.linspace(-np.ceil(pull_max), np.ceil(pull_max) , 3))
    else:
        axis.set_ylim(-3.5, 3.5)
        axis.set_yticks([-3,0,3])    
    axis.set_xlim(limits)
    axis.set_ylabel('Pull', loc='center')
    axis.set_xlabel(xlabel)
    
    mask_ = h[0]>0
    denominator = h[0][mask_]
    #denominator = expected_events[mask_]
    if return_chi2:
        if return_expected_evts:    
            return np.sum(np.power(h[0][mask_]-expected_events[mask_],2)/denominator), expected_events
        return np.sum(np.power(h[0][mask_]-expected_events[mask_],2)/denominator)
    elif return_expected_evts:
        return expected_events
    

    

def findPdfBySubString(model, substring):
    for pdf in model.pdfs:
        if substring in pdf.name: return pdf
def findFracBySubString(model, substring):
    for indx, pdf in enumerate(model.pdfs):
        if substring in pdf.name: return model.fracs[indx]
    
def plot_projection(data, model, var_to_integrate, axis, bins=70,  return_chi2 = False, 
                   pulls=False, axis_pulls=None, plot_components=False,
                   print_params=False, print_chi2_dof=True, params_text_opts = dict(), **kwargs):

    pdf = model.create_projection_pdf(var_to_integrate)
    
    h = plot_simple_model(data, pdf, axis, bins, pdf_name='Projection', **kwargs)
    signal_pdf = findPdfBySubString(model, 'Signal')
    s_frac     = findFracBySubString(model, 'Signal')
    background_pdf = findPdfBySubString(model, 'Backg')
    b_frac     = findFracBySubString(model, 'Back')
    
    plot_components_spec(data, pdf,  signal_pdf, s_frac,
                         var_to_integrate, 0,
                         axis, bins, **kwargs)
    plot_components_spec(data, pdf,  background_pdf, b_frac,
                         var_to_integrate, 1,
                         axis, bins, **kwargs)

    axis.legend(loc=kwargs.get('label_pos', 'best'),
                fontsize=kwargs.get('label_size', 18), 
                ncol=2,  frameon=True)
    axis.set_ylim(0, max(h[0])*1.2)
    
    if pulls and not axis_pulls:
        raise NotFoundError('You need to pass another axis for the pulls ')
    elif pulls:
        axis.set_xticks([])
        print(pdf.obs[0], pdf.norm_range.limit1d)
        chi2 = plot_pull(h, pdf, pdf.obs[0], axis_pulls, return_chi2=True)
        if print_chi2_dof:
            dof_int = bins-len(pdf.params) if not print_params else bins-len(print_params.params)
            dof_int -=1
            #print(chi2)
            tex_chi = r'$ \chi^2 /DOF$ = ' +f'{round(chi2,3)}/{dof_int} = {round(chi2/dof_int,3)}'
            chi_x, chi_y = kwargs.get('chi_x', 0.5),  kwargs.get('chi_y', 0.5)
            axis.text( chi_x, chi_y , tex_chi, va='bottom',
                      fontsize=kwargs.get('fontsize', 18), transform = axis.transAxes)
        return h, chi2
    else:
        return h
    
def plot_components_spec(data, projection, model, frac,
                         var_to_integrate, style_indx,
                         axis, bins, levels=1, **kwargs):
    
    limits = projection.norm_range.limit1d
    h = np.histogram(data, bins=bins, range=limits)
    bin_mean = (h[1][1:]+h[1][:-1])/2
    bin_size = h[1][1]-h[1][0]
    n_events = np.sum(h[0])
    scale = bin_size*n_events
    y_err = np.sqrt(h[0]*(1-h[0]/n_events))
    mask_ = h[0]>0

    hatces     = ['//', '\\', '--']
    facecolors = ['lightcoral', 'lightblue', 'palegreen']
    edgecolors = ['orangered' , 'dodgerblue', 'darkgreen']
    zorders    = [50,20,5] 
    x = np.linspace(*limits, 1000)
    line_colors = ['darkred', 'navy']
    line_styles = ['--', ':', '-.']
    
    
    for component in model.pdfs:
        print(style_indx, component.name)
        if (len(component.obs)>1) or (component.obs==var_to_integrate.obs) : continue
        print(style_indx, component.name)
        #Primer Plottemos la componente  de la pdf total
        # signal, background, etc
        #name = ''.join(c.title() for c in component.name.split('__', ' '))
        name =component.name
        axis.fill_between(x, component.pdf(x)*scale*frac, alpha=0.6,
                facecolor=facecolors[style_indx], hatch=hatces[style_indx], 
                edgecolor=edgecolors[style_indx], label=name,
                zorder = zorders[style_indx])
        try:
            #if str(type(component))=="<class 'zfit.models.functor.ProductPDF'>": 
            #        raise NotImplementedError
            for indx, part in enumerate(component.pdfs):
                try:
                    frac__ = component.fracs[indx]
                except Exception as e:
                    print(e)
                    frac__=1
                    
                if len(part.obs)>1 or (part.obs==var_to_integrate.obs) : continue
                axis.plot(x, part.pdf(x)*scale*frac*frac__, 
                          color=edgecolors[style_indx],
                          ls = line_styles[indx],
                          linewidth=3,
                          label=part.name,
                          zorder=2000
                         )
        except Exception as e:
            print(e)
        
    
def plot_simple_model(data, pdf, axis, bins=70, **kwargs):
    limits = pdf.norm_range.limit1d
    h = np.histogram(data, bins=bins, range=limits)
    bin_mean = (h[1][1:]+h[1][:-1])/2
    bin_size = h[1][1]-h[1][0]
    n_events = np.sum(h[0])
    scale = bin_size*n_events
    y_err = np.sqrt(h[0]*(1-h[0]/n_events))
    mask_ = h[0]>0
    
    axis.errorbar(bin_mean[mask_], h[0][mask_], xerr=bin_size/2, yerr=y_err[mask_], ms=20,
                label='Data', ls='none', capsize=2, color='black')
    
    
    x = np.linspace(limits[0], limits[1], 1000)
    model_name = kwargs.get('pdf_name', pdf.name)
    axis.plot(x, pdf.pdf(x)*scale, linewidth=3, zorder=1000,
            ls='-', color=kwargs.get('MainColor', 'black'), label=model_name)
    axis.set_xlim(limits)
    return h
    
    
def model_has_pdfs(model):
    try:
        model.pdfs
        return True
    except AttributeError:
        return False
    
def model_has_fracs(model):
    try:
        model.fracs
        return True
    except AttributeError:
        return False
    

def bin_model(model, bins=20, integrate=False, verbose=True, center=True):

    # Find out if bins is a list or an integer
    # Missing escalating when model is extended
    if type(bins) in [list, np.ndarray]:
        h1 = np.array(bins)
    else:
        limits = model.norm_range.limit1d
        step = (limits[1]-limits[0])/bins
        h1   = np.array([limits[0]+i*step for i in range(bins+1)])
    
    binned = np.zeros(len(h1)-1, dtype=np.float128)
    if integrate:
        for i in range(len(h1)-1):
            integration = model.integrate(limits=[h1[i],h1[i+1]]).numpy()[0]
            binned[i] = integration
            if verbose : print(f'   Bin {i}, {integration}')
    else:
        if center:
            bin_center = (h1[1:]+h1[:-1])/2
            evaluate = model.pdf(bin_center).numpy()
            binned = evaluate*(h1[1:]-h1[:-1])
        else:
            evaluate = model.pdf(h1).numpy()
            binned = (evaluate[1:]+evaluate[:-1])*(h1[1:]-h1[:-1])/2

    return binned

def plot_model(data, 
               pdf,
               axis=None, 
               bins=70,  
               return_chi2 = False, 
               weights='none',
               pulls=False, 
               axis_pulls=None, 
               plot_components=False,
               print_params=False, 
               print_chi2_dof=True, 
               params_text_opts = dict(x=0.6, y=0.4, ncol=1, fontsize=15), 
               remove_string='',
               main_kwargs=dict(),
               data_kwargs=dict(capsize=2, 
                                color='black', 
                                ms=20),
               chi_x = 0.5, 
               chi_y = 0.5,
               level=1,
               return_expected_evts = False,
               regex='',
               **kwargs):
    """ Tries to be an all-in-one 1D-plotting for (~kind of) HEP style.
        Can create pulls given a binning, and also evaluate chi2/DOF
            - where DOF = (nbins-1)-params
        Also can print the fitted params with its error as given by a 
            zfit.minimizers.fitresult.FitResult
        It incorporates many dictionaries to customize the settings.
    
    Parameters
    ---------------
    data: pd.DataFrame, list, array, Iterable
        Data to be compared against a model
    pdf: zfit.models
        Any instance of a ZFIT model, if it has 
    axis: matplotlib.axes
        An axis to be plotted the figure. 
            Plotting with no axis is TO BE IMPLEMENTED
    main_kwargs: dict
        Arguments to be passed to the Top model,
        dict(fill=True) : create a fill plot instead of a simple plot 
    Returns
    ---------------
    Data histogram: (np.array, np.array)
        Output from np.histogram of Data
    chi2: float
        chi2 evaluated from the binning and taking into account bins with counts>0
    """
    if not axis:
        fig,axis = plt.subplots()
    try:
        pdf.norm_range.spaces
        limits = [pdf.norm_range.spaces[0].lower.flatten()[0],
                  pdf.norm_range.spaces[-1].upper.flatten()[0]
        ]
    except AttributeError:
        limits = pdf.norm_range.limit1d
    if np.all(weights=='none'):
        weights = np.ones_like(data)
    h = np.histogram(data, bins=bins, range=limits, weights=weights)
    bin_mean = (h[1][1:]+h[1][:-1])/2
    bin_size = h[1][1]-h[1][0]
    n_events = np.sum(h[0])
    #scale_ = bin_size*n_events
    scale = np.sum(np.diff(h[1])*h[0])
    y_err = np.sqrt(h[0]*(1-h[0]/n_events))
    mask_ = h[0]>0
    
    axis.errorbar(bin_mean[mask_], 
                  h[0][mask_], 
                  xerr=bin_size/2, 
                  yerr=y_err[mask_], 
                  label='Data', 
                  ls='none', 
                  **data_kwargs
                 )
    

    main_kw = deepcopy(main_kwargs) 
    if kwargs.get('MainColor', False):
        print(main_kw)
        if 'color' in main_kw:
            raise ValueError('color (MainColor) already specified')
        else:
            main_kw['color'] = kwargs['MainColor']
            

    x = np.linspace(limits[0], limits[1], 1000)
    model_name = kwargs.get('pdf_name', pdf.name)
    if 'fill' in main_kw:
        del main_kw['fill']
        axis.fill(x, pdf.pdf(x)*scale, zorder=1000,
            **main_kw, label=model_name)
    else:
        axis.plot(x, pdf.pdf(x)*scale, zorder=1000,
            **main_kw, label=model_name)
    
    #if type(pdf) != zfit.pdf.SumPDF and plot_components:
    #    print('NOT COMPONENTS')
    try: 
        n_models = len(pdf.models)
    except Exception as e: 
        print(e)
        n_models = -1
    if plot_components and n_models <= 0:
        print('NOT COMPONENTS')
        raise NotImplementedError('PDF has no componets')
        
    elif plot_components:
        hatces     = ['', '', '--']
        facecolors = ['lightcoral', 'lightblue', 'palegreen']
        edgecolors = ['orangered' , 'dodgerblue', 'darkgreen']
        zorders    = [50,20,5] 
        for i in range(len(pdf.pdfs)):
            model = pdf.pdfs[i]
            if model_has_fracs(pdf): frac = pdf.fracs[i]
            else: frac = 1
            if 'decay' in model.name.lower(): name = 'Angular Signal'
            else : name=model.name.replace('_extended', '')
            axis.fill_between(x, model.pdf(x)*scale*frac, alpha=0.6,
                    facecolor=facecolors[i], hatch=hatces[i], 
                    edgecolor=edgecolors[i], label=name,
                    zorder = zorders[i])
            #level=2
            if model_has_pdfs(model) and level==2 and regex in model.name.lower():
                ls = ['--', ':', '-.', '--', ':', '.-']
                linewidths = [1.5, 1.5, 1.5, 3, 3, 3,]
                #print(model.name, 'MODELS')
                for j in range(len(model.pdfs)):
                    submodel = model.pdfs[j]
                    if model_has_fracs(model): frac_ = model.fracs[j]
                    else: frac_ = 1
                    if 'decay' in submodel.name.lower(): name_ = 'Angular Signal'
                    else : name_=submodel.name.replace('_extended', '')
                    axis.plot(x, submodel.pdf(x)*scale*frac*frac_, 
                                ls=ls[j],
                                linewidth=linewidths[j],
                                color=edgecolors[i], 
                                label=name_.replace(remove_string, ' ').replace('  ', ' '),
                                zorder = zorders[i]*10)

    
    if print_params:
        
        def get_opt_indx(_, i):
            if _ is None: _ = 0.5
            elif type(_)==float: _=_
            elif len(_)==1: _ = _[0]
            else: _ = _[i]
            return _
        if type(print_params)==bool:
            texts = textParams_from_model(pdf, params_text_opts.get('ncol', 2))
        else:
            texts = textParams(print_params, params_text_opts.get('ncol', 2))
        print(texts)
        x = params_text_opts.get('x',None)
        y = params_text_opts.get('y',None)
        if 'x' in params_text_opts: del params_text_opts['x']
        if 'y' in params_text_opts: del params_text_opts['y']
        if 'ncol' in params_text_opts: del params_text_opts['ncol']
        
        for i, text in enumerate(texts):
            if remove_string: text = text.replace(remove_string, '')
            x_ = get_opt_indx(x, i)
            y_ = get_opt_indx(y, i)
            axis.text(x_, y_, text, 
                      transform = axis.transAxes, **params_text_opts
                     )

    
    if 'log' in kwargs:
        axis.set_yscale('log')
        axis.set_ylim(0, np.max(h[0])*20)
    else:
        axis.set_ylim(0, np.max(h[0])*1.3)
    axis.set_xlim(limits)
    axis.set_ylabel('Events / '+str(round(bin_size, kwargs.get('round_binsz', 4))))
    axis.legend(fontsize=kwargs.get('fontsize', 12), 
                loc=kwargs.get('loc', 1), 
                frameon=kwargs.get('frameon', True), 
                framealpha=kwargs.get('framealpha', 0.8), 
                ncol=kwargs.get('ncol', 1))
    model_obs = pdf.obs
    if type(model_obs)==tuple : model_obs = model_obs[0]
    xlabel = kwargs.get('xlabel', model_obs)
    if not pulls or not axis_pulls:
        axis.set_xlabel(xlabel)
    else:
        axis.set_xticklabels([])
    
    #print('formationg')
    #axis.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
    
    
    if pulls and not axis_pulls:
        raise NotFoundError('You need to pass another axis for the pulls ')
    elif pulls:
        chi2 = plot_pull(h, pdf, xlabel, axis_pulls, return_chi2=True, return_expected_evts=return_expected_evts)
        if type(chi2) in [list, tuple] : chi2, expected_events = chi2
        if print_chi2_dof:
            n_params = len(pdf.params)
            if print_params and type(print_params)!=bool:
                n_params = len(print_params.params)
            dof_int = bins-n_params
            dof_int -=1
            #print(chi2)
            tex_chi = r'$ \chi^2 /DOF$ = ' +f'{round(chi2,3)}/{dof_int} = {round(chi2/dof_int,3)}'
            #chi_x, chi_y = kwargs.get('chi_x', 0.5),  kwargs.get('chi_y', 0.5)
            axis.text( chi_x, chi_y , tex_chi, va='bottom', ha=kwargs.get('ha_chi', 'left'),
                      fontsize=kwargs.get('fontsize_chi2', 18), zorder=kwargs.get('chi_zorder', 100), transform = axis.transAxes)
        if return_expected_evts:
            return h, chi2, expected_events
        else:       
            return h, chi2
    else:
        if return_chi2:
            mask_ = h[0]>0
            binned_pdf = bin_model(pdf, h[1])
            expected_events = binned_pdf*n_events
            #chi2 = np.sum(np.power(h[0][mask_]-expected_events[mask_],2)/expected_events[mask_])
            chi2 = np.sum(np.power(h[0][mask_]-expected_events[mask_],2)/h[0][mask_])
            return h, chi2
        return h 
    
    

    
    
def plot_models(data, 
               pdfs,
               label_names,
               axis=None, 
               bins=70,  
               return_chi2 = False, 
               weights='none',
               pulls=False, 
               axis_pulls=None, 
               plot_components=False,
               print_params=False, 
               print_chi2_dof=True, 
               params_text_opts = dict(x=0.6, y=0.4, ncol=1, fontsize=15), 
               remove_string='',
               main_kwargs=dict(),
               data_kwargs=dict(capsize=2, 
                                color='black', 
                                ms=20),
               chi_x = 0.5, 
               chi_y = 0.5,
               level=1,
               return_expected_evts = False,
               regex='',
               **kwargs):
    """ Tries to be an all-in-one 1D-plotting for (~kind of) HEP style.
        Can create pulls given a binning, and also evaluate chi2/DOF
            - where DOF = (nbins-1)-params
        Also can print the fitted params with its error as given by a 
            zfit.minimizers.fitresult.FitResult
        It incorporates many dictionaries to customize the settings.
    
    Parameters
    ---------------
    data: pd.DataFrame, list, array, Iterable
        Data to be compared against a model
    pdf: zfit.models
        Any instance of a ZFIT model, if it has 
    axis: matplotlib.axes
        An axis to be plotted the figure. 
            Plotting with no axis is TO BE IMPLEMENTED
    main_kwargs: dict
        Arguments to be passed to the Top model,
        dict(fill=True) : create a fill plot instead of a simple plot 
    Returns
    ---------------
    Data histogram: (np.array, np.array)
        Output from np.histogram of Data
    chi2: float
        chi2 evaluated from the binning and taking into account bins with counts>0
    """
    if not axis:
        fig,axis = plt.subplots()
    try:
        pdfs[0].norm_range.spaces
        limits = [pdfs[0].norm_range.spaces[0].lower.flatten()[0],
                  pdfs[0].norm_range.spaces[-1].upper.flatten()[0]
        ]
    except AttributeError:
        limits = pdfs[0].norm_range.limit1d
    if np.all(weights=='none'):
        weights = np.ones_like(data)
    h = np.histogram(data, bins=bins, range=limits, weights=weights)
    bin_mean = (h[1][1:]+h[1][:-1])/2
    bin_size = h[1][1]-h[1][0]
    n_events = np.sum(h[0])
    #scale_ = bin_size*n_events
    scale = np.sum(np.diff(h[1])*h[0])
    y_err = np.sqrt(h[0]*(1-h[0]/n_events))
    mask_ = h[0]>0
    
    axis.errorbar(bin_mean[mask_], 
                  h[0][mask_], 
                  xerr=bin_size/2, 
                  yerr=y_err[mask_], 
                  label='Data', 
                  ls='none', 
                  **data_kwargs
                 )
    

    main_kw = deepcopy(main_kwargs) 
    if kwargs.get('MainColor', False):
        print(main_kw)
        if 'color' in main_kw:
            raise ValueError('color (MainColor) already specified')
        else:
            main_kw['color'] = kwargs['MainColor']
            
    # Here is where the pdf magic happens

    x = np.linspace(limits[0], limits[1], 1000)
    for idx, pdf in enumerate(pdfs):
        model_name = kwargs.get('pdf_name', pdf.name)
        if 'fill' in main_kw:
            del main_kw['fill']
            axis.fill(x, pdf.pdf(x)*scale, zorder=1000,
                **main_kw, label=label_names[idx])
        else:
            axis.plot(x, pdf.pdf(x)*scale, zorder=1000,
                **main_kw,  label=label_names[idx])
    
        #if type(pdf) != zfit.pdf.SumPDF and plot_components:
        #    print('NOT COMPONENTS')
        try: 
            n_models = len(pdf.models)
        except Exception as e: 
            print(e)
            n_models = -1
        if plot_components and n_models <= 0:
            print('NOT COMPONENTS')
            raise NotImplementedError('PDF has no componets')

        elif plot_components:
            hatces     = ['', '', '--']
            facecolors = ['lightcoral', 'lightblue', 'palegreen']
            edgecolors = ['orangered' , 'dodgerblue', 'darkgreen']
            zorders    = [50,20,5] 
            for i in range(len(pdf.pdfs)):
                model = pdf.pdfs[i]
                if model_has_fracs(pdf): frac = pdf.fracs[i]
                else: frac = 1
                if 'decay' in model.name.lower(): name = 'Angular Signal'
                else : name=model.name.replace('_extended', '')
                axis.fill_between(x, model.pdf(x)*scale*frac, alpha=0.6,
                        facecolor=facecolors[i], hatch=hatces[i], 
                        edgecolor=edgecolors[i], label=label_names[idx],
                        zorder = zorders[i])
                #level=2
                if model_has_pdfs(model) and level==2 and regex in model.name.lower():
                    ls = ['--', ':', '-.', '--', ':', '.-']
                    linewidths = [1.5, 1.5, 1.5, 3, 3, 3,]
                    #print(model.name, 'MODELS')
                    for j in range(len(model.pdfs)):
                        submodel = model.pdfs[j]
                        if model_has_fracs(model): frac_ = model.fracs[j]
                        else: frac_ = 1
                        if 'decay' in submodel.name.lower(): name_ = 'Angular Signal'
                        else : name_=submodel.name.replace('_extended', '')
                        axis.plot(x, submodel.pdf(x)*scale*frac*frac_, 
                                    ls=ls[j],
                                    linewidth=linewidths[j],
                                    color=edgecolors[i], 
                                    label=name_.replace(remove_string, ' ').replace('  ', ' '),
                                    zorder = zorders[i]*10)

    
        if print_params:

            def get_opt_indx(_, i):
                if _ is None: _ = 0.5
                elif type(_)==float: _=_
                elif len(_)==1: _ = _[0]
                else: _ = _[i]
                return _
            if type(print_params)==bool:
                texts = textParams_from_model(pdf, params_text_opts.get('ncol', 2))
            else:
                texts = textParams(print_params, params_text_opts.get('ncol', 2))
            print(texts)
            x = params_text_opts.get('x',None)
            y = params_text_opts.get('y',None)
            if 'x' in params_text_opts: del params_text_opts['x']
            if 'y' in params_text_opts: del params_text_opts['y']
            if 'ncol' in params_text_opts: del params_text_opts['ncol']

            for i, text in enumerate(texts):
                if remove_string: text = text.replace(remove_string, '')
                x_ = get_opt_indx(x, i)
                y_ = get_opt_indx(y, i)
                axis.text(x_, y_, text, 
                          transform = axis.transAxes, **params_text_opts
                         )

    
    # if 'log' in kwargs:
    #     axis.set_yscale('log')
    #     axis.set_ylim(0, np.max(h[0])*20)
    # else:
    #     axis.set_ylim(0, np.max(h[0])*1.3)
    # axis.set_xlim(limits)
    # axis.set_ylabel('Events / '+str(round(bin_size, kwargs.get('round_binsz', 4))))
    # axis.legend(fontsize=kwargs.get('fontsize', 12), 
    #             loc=kwargs.get('loc', 1), 
    #             frameon=kwargs.get('frameon', True), 
    #             framealpha=kwargs.get('framealpha', 0.8), 
    #             ncol=kwargs.get('ncol', 1))
    # model_obs = pdf.obs
    # if type(model_obs)==tuple : model_obs = model_obs[0]
    # xlabel = kwargs.get('xlabel', model_obs)
    # if not pulls or not axis_pulls:
    #     axis.set_xlabel(xlabel)
    # else:
    #     axis.set_xticklabels([])
    
    #print('formationg')
    #axis.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
    
    
    # if pulls and not axis_pulls:
    #     raise NotFoundError('You need to pass another axis for the pulls ')
    # elif pulls:
    #     chi2 = plot_pull(h, pdf, xlabel, axis_pulls, return_chi2=True, return_expected_evts=return_expected_evts)
    #     if type(chi2) in [list, tuple] : chi2, expected_events = chi2
    #     if print_chi2_dof:
    #         n_params = len(pdf.params)
    #         if print_params and type(print_params)!=bool:
    #             n_params = len(print_params.params)
    #         dof_int = bins-n_params
    #         dof_int -=1
    #         #print(chi2)
    #         tex_chi = r'$ \chi^2 /DOF$ = ' +f'{round(chi2,3)}/{dof_int} = {round(chi2/dof_int,3)}'
    #         #chi_x, chi_y = kwargs.get('chi_x', 0.5),  kwargs.get('chi_y', 0.5)
    #         axis.text( chi_x, chi_y , tex_chi, va='bottom', ha=kwargs.get('ha_chi', 'left'),
    #                   fontsize=kwargs.get('fontsize_chi2', 18), zorder=kwargs.get('chi_zorder', 100), transform = axis.transAxes)
    #     if return_expected_evts:
    #         return h, chi2, expected_events
    #     else:       
    #         return h, chi2
    # else:
    #     if return_chi2:
    #         mask_ = h[0]>0
    #         binned_pdf = bin_model(pdf, h[1])
    #         expected_events = binned_pdf*n_events
    #         #chi2 = np.sum(np.power(h[0][mask_]-expected_events[mask_],2)/expected_events[mask_])
    #         chi2 = np.sum(np.power(h[0][mask_]-expected_events[mask_],2)/h[0][mask_])
    #         return h, chi2
    #     return h
    
    
    
    
    
    
def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5, label=''):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror, yerror):
        rect = Rectangle((x - xe, y - ye), 2*xe, 2*ye)
        errorboxes.append(rect)

    # Create patch collection with specified boxes
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, label=label)

    # Add collection to axes
    ax.add_collection(pc)


    #return artists

def data_for_error_boxes(minimums, param):
    y, yerr = list(), list()
    for k in range(-1,9):
        print(k)
        if k in [3,5]:continue
        for k, err in minimums[k].params.items():
            if param in k.name:
                y.append(k.value())
                yerr.append(err['hesse_np']['error'])
    return y, yerr



def plot_measurement_q2(mean, error, bins, q2_width, q2_mean, axis, poi='afb', ylims=None, color='r', label='', only_boxes=False, **kwargs):
        
    if not ylims:
        sum_ = [mean[i]+error[i] for i in range(len(mean))]
        dif_ = [mean[i]-error[i] for i in range(len(mean))]
        ymax = max(sum_)
        ymin = min(dif_)
    else:
        ymax=ylims[1]
        ymin=ylims[0]
        
    make_error_boxes(axis, q2_mean, mean, q2_width, error, facecolor=color, label=label )
    
    
    axis.errorbar(q2_mean, mean, xerr=q2_width, yerr=error,
                 ls='none', capsize=5, color=color,
                 alpha=kwargs.get('alpha', 0.5)+0.1)
    
    if 'scatter_size' in kwargs:
        axis.scatter(q2_mean, mean, color=color, 
                 alpha=kwargs.get('alpha', 0.5)+0.2, 
                 label=label, s = kwargs['scatter_size'])
    else:
        axis.scatter(q2_mean, mean, color=color, 
                 alpha=kwargs.get('alpha', 0.5)+0.2, 
                 label=label)
    
    if not only_boxes:
        print(ymin, ymax)
        border = (ymax-ymin)/10
        times=1.3
        r_max = 1.3
        
        
        n = 10

        axis.fill_betweenx([ymin-n*border, ymax+n*border], 
                           [bins['3'][0],bins['3'][0]], 
                           [bins['3'][1],bins['3'][1]], 
                           facecolor='lightgrey', 
                           edgecolor='silver',
                           zorder=20,
                           #hatch='xx',
                          alpha=0.5)
        """
        axis.axis.plot([bins['3'][0],
                   bins['3'][0]],
                  [ymin-n*border, ymax+n*border], 
                  ls='--', 
                  color='grey',
                  zorder=30)
        axis.plot([bins['3'][1],
                   bins['3'][1]],
                  [ymin-n*border, ymax+n*border], 
                  ls='--', 
                  color='grey', zorder=30)
        """
        #axis.vlines(bins['3'], ymin-n*border, ymax+n*border,
        #            color='grey', zorder=30)
        """
                axis.fill_betweenx([ymin-n*border, ymax+n*border], [bins['5'][0],bins['5'][0]], [bins['5'][1],bins['5'][1]], facecolor='none', hatch='xx', edgecolor='silver', zorder=20)
                axis.plot([bins['5'][0],bins['5'][0]],[ymin-n*border, ymax+n*border], ls='--', color='black', zorder=30)
                axis.plot([bins['5'][1],bins['5'][1]],[ymin-n*border, ymax+n*border], ls='--', color='black', zorder=30)
        """
        axis.fill_betweenx([ymin-n*border, ymax+n*border], 
                           [bins['5'][0],bins['5'][0]], 
                           [bins['5'][1],bins['5'][1]], 
                           facecolor='lightgrey', 
                           edgecolor='silver',
                           zorder=20,
                           #hatch='xx',
                          alpha=0.5)

        axis.set_ylim(ymin-border, ymax+border)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
#///////
    
def compare_plot(Data_Num,
                 Data_Den,
                 weights_Num=None,
                 weights_Den=None,
                 label_Num = '',
                 label_Den = '',
                 title='',
                 operation='ratio',
                 hist_opts=dict(bins=50), 
                 density=True,
                 opts_commons  = dict(),
                 opts_Num_plot = dict(),
                 opts_Den_plot = dict(),
                 opts_lower_plot = dict(),
                 low_ylabel='',
                 low_xlabel='',
                 low_ylim = [0,2],
                 ylim = None,
                 ks_t  = 'cut', 
                 out_dir=None,
                 out_name='',
                 axes = [None, None],
                 return_axis=False,
                 show=False,
                 return_k_val=False,
                 lower_lines=True,
                 ):
    """Plot two samples as histograms with same binning and evaluate the ratio of their hieghts,
    if both samples came from the distribution the ratio should be distributied uniformly
    
    Params:
    ks_t = bool, str
        If True, evaluate the weighted KS test with the complete samples
        If 'cut', evaluate the weighted KS test with a sub-sample as seen in the plot
    operation = str,
        Valid opts: ratio, difference
    """
    
    
    if all(axes):
        _main, _lower = axes
        #_main.set_title(title)
    else:
        fig = plt.figure()
        fig.suptitle(title, y=0.93)
        _main, _lower = create_axes_for_pulls(fig)
    

    
    Histo_Num = hist_weighted(Data_Num, 
                              **hist_opts,
                              weights=weights_Num, 
                              axis=_main, 
                              density=density,
                              label = label_Num,
                             **opts_Num_plot,
                             **opts_commons)
    
    Histo_Den = hist_weighted(Data_Den, 
                              bins=Histo_Num[1],
                              weights=weights_Den, 
                              axis=_main, 
                              density=density,
                              label = label_Den,
                             **opts_Den_plot,
                             **opts_commons)
    
    
    if ks_t=='cut':
        low_cut = np.max([
                    Histo_Num[1][0],
                    np.percentile(Data_Num, 0.1),
                    np.percentile(Data_Den, 0.1) ])
        upp_cut = np.min([
                    Histo_Num[1][-1],
                    np.percentile(Data_Num, 99.9),
                    np.percentile(Data_Den, 99.9) ])
        print(low_cut, upp_cut)
        """
        low_cut = Histo_Num[1][0] \
                 if Histo_Num[1][0]>np.percentile(Data_Num, 0.1) \
                 else np.percentile(Data_Num, 0.1)
        upp_cut = Histo_Num[1][-1] \
                  if Histo_Num[1][-1]<np.percentile(Data_Den, 99.9) else np.percentile(Data_Den, 99.9)"""
        ks_ = ks_test.ks_2samp_weighted(
            Data_Num[(Data_Num>=low_cut) & (Data_Num<=upp_cut)],
            Data_Den[(Data_Den>=low_cut) & (Data_Den<=upp_cut)],
            weights_Num[(Data_Num>=low_cut) & (Data_Num<=upp_cut)] \
                                                     if np.all(weights_Num) else None,
            weights_Den[(Data_Den>=low_cut) & (Data_Den<=upp_cut)] \
                                                    if np.all(weights_Den) else None,)
    elif ks_t:
        ks_ = ks_test.ks_2samp_weighted(
            Data_Num,    Data_Den,
            weights_Num, weights_Den )
    else:
        ks_ = None
    label_title = 'KS $p_{val}$ = '+ str(round(ks_[1], 4)) if ks_ else None
    if label_Den or label_Num:
        if axes[0] and axes[0].get_legend():
            previous_title = axes[0].get_legend().get_title().get_text()
            if previous_title: 
                label_title = previous_title+'\n'+label_title
            
        _main.legend(frameon=True, title=label_title, fontsize=18)
    if ylim=='zero':
        _main.set_ylim(ymin=0)
    elif ylim:
        _main.set_ylim(*ylim)
        
    bin_mean = (Histo_Num[1][1:]+Histo_Num[1][:-1])/2
    bin_size = (bin_mean[1]-bin_mean[0])/2
    r = int(abs(np.log10(bin_size)))+2
    if density:
        _main.set_ylabel(f'Density / {str(round(bin_size*2,r))[:r+1]}')
    else:
        _main.set_ylabel(f'Counts / {str(round(bin_size*2,r))[:r+1]}')
    
    if operation=='ratio':
        ratio = Histo_Num[0]/Histo_Den[0]
        ratio = np.where(np.isnan(ratio), np.inf, ratio)
        scale_sum_ratio = np.sum(Histo_Num[0])/np.sum(Histo_Den[0])
        error = ratio*np.hypot(Histo_Num[-1]/Histo_Num[0], Histo_Den[-1]/Histo_Den[0])
    
    elif operation=='difference':
        ratio = Histo_Num[0]-Histo_Den[0]
        ratio = np.where(np.isnan(ratio), np.inf, ratio)
        scale_sum_ratio = np.mean(Histo_Num[0]-Histo_Den[0])
        error = np.hypot(Histo_Num[-1], Histo_Den[-1])

    # Error bar used to handle nans and infs quite well (ignoring them), 
    # now we have to make sure they are finite numbers, at least.
    finite_mask = np.isfinite(ratio)
    _lower.errorbar(bin_mean[finite_mask], 
                    ratio[finite_mask], 
                    xerr=bin_size, 
                    yerr=error[finite_mask], 
                    **opts_lower_plot)
    
    if type(lower_lines)==str and (lower_lines.lower()=='mean' or lower_lines.lower()=='average'):
        _lower.axhline(scale_sum_ratio,     ls=':',  color='grey')
    elif lower_lines:
        _lower.axhline(0.5*scale_sum_ratio, ls='--', color='grey', alpha=0.75)
        _lower.axhline(1.5*scale_sum_ratio, ls='--', color='grey', alpha=0.75)
        _lower.axhline(scale_sum_ratio,     ls=':',  color='grey')
    _lower.set_ylabel(low_ylabel, fontsize=15, loc='center')
    _lower.set_xlabel(low_xlabel)
    
    if low_ylim and density==False:
        print(scale_sum_ratio)
        _lower.set_ylim(low_ylim[0]*scale_sum_ratio, low_ylim[1]*scale_sum_ratio)
    elif low_ylim:
        _lower.set_ylim(*low_ylim)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{k}{out_name}.png'),
                    bbox_inches='tight',
                    dpi=100
                   )
        
    to_return = list()
    
    if return_axis:
        to_return+=[fig, (_main, _lower)]
    
    if return_k_val:
        to_return.append(ks_[1])
        
    if show:
        plt.show()
    
    if to_return:
        if len(to_return)==1: return to_return[0]
        return to_return
        




def plot_correlation(minimum, replace_str='', title='', figsize=(12, 12)):

    plt.figure(figsize=figsize)
    if title: plt.title(title)
    names = [p.name.strip().replace(replace_str, '') for p in minimum.params.keys()]
    
    hmap = sns.heatmap(minimum.correlation(), 
                vmin=-1, 
                vmax=1, 
                cmap='seismic', 
                annot=True, 
                annot_kws=dict(fontsize=15), 
                xticklabels=names, 
                yticklabels=names, 
                cbar_kws=dict(label='Correlation')
                      )
    hmap.xaxis.set_ticks_position('none') 
    hmap.yaxis.set_ticks_position('none') 

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    for i in range(len(minimum.correlation())):
        hmap.axhline(i, color='black', linewidth=0.5)
    for i in range(len(minimum.correlation().T)):
        hmap.axvline(i, color='black', linewidth=0.5)