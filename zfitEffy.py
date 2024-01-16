import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import json
plt.style.use(hep.style.CMS)
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import zfit
from scipy.special import binom
import sys
import tools
import plot_tools
from customPDFs import bernstein, truncated_bernstein, non_negative_chebyshev
from random import random
import pdb
from scipy import stats



def get_f_test(RSS1, n_param1, RSS2, n_param2, nbins):
    dof1 = n_param2-n_param1
    dof2 = nbins-n_param2
    numerator = (RSS1-RSS2)/dof1
    denominator = RSS2/dof2
    F_statistic = numerator/denominator
    F_pval = 1-stats.f.cdf(float(F_statistic), dof1, dof2 )
    return float(F_statistic), float(F_pval)


def create_non_neg_chebyshev_models(
                obs,
                max_degree, 
                positive_constraint=False, 
                rand=False, 
                name='',
                fixed_coef=-1,
                ):

    models = dict()
    def ini_val(rand, indx):
        if rand: ini = random()
        else:
            ini = 0.1
            if indx==0: ini=1
        return ini    
    
    for j in range(0, max_degree):

        if positive_constraint:
            coefsR = [zfit.Parameter(f'{name}c^{i}_{j}', ini_val(rand, i), 0, 10, 0.001) 
            if  fixed_coef!=i else 1 for i in range(j+1)]
        else:
            coefsR = [zfit.Parameter(f'{name}c^{i}_{j}', value=ini_val(rand, i))  
            if  fixed_coef!=i else 1 for i in range(j+1)]
        models[j] = non_negative_chebyshev(coefsR, obs)
    return models
    
    
    
    

def create_chebyshev_models(
                obs,
                max_degree, 
                positive_constraint=False, 
                rand=False, 
                name='',
                coeff0=True,
                low = None ,
                up = None,
                step = 0.000001
                ):

    models = dict()
    def ini_val(rand):
        if rand: ini = random()
        else: ini = 0.1
        return ini    
    
    for j in range(0, max_degree):
        if positive_constraint:
            coefsR = [zfit.Parameter(f'{name}c^{i}_{j}', 
                                        ini_val(rand), 
                                        0,
                                        10 if not up else up,
                                        step)  
                     for i in range(1,j+1)]
        else:
            coefsR = [zfit.Parameter(f'{name}c^{i}_{j}', 
                                     ini_val(rand),
                                     low,
                                     up,
                                     step
                                     )  
                      for i in range(1,j+1)]
        if coeff0 and positive_constraint:
            coeff0_ = zfit.Parameter(f'{name}c^0_{j}', 
                                        1, 
                                        0, 
                                        10 if not up else up,
                                        step)
        elif coeff0 and not positive_constraint:
            coeff0_ = zfit.Parameter(f'{name}c^0_{j}', 
                                     value=1)
        else:
            coeff0_ = None
        
        models[j] = zfit.pdf.Chebyshev(obs, coefsR, coeff0=coeff0_)
        
    return models
    
    
    
def create_bernstein_models(obs, max_degree, 
    positive_constraint=True, 
    rand=False, 
    name='', 
    truncated=False,
    low = None ,
    up = 500,
    step = 0.00001 ,
    fixed_coef=-1):

    models = dict()
    
    def ini_val(rand):
        if rand:
            ini = random()
        else:
            ini = 0.1
        return ini
    
    
    for j in range(0, max_degree):

        if positive_constraint:
            coefsR = [zfit.Parameter(f'{name}c^{i}_{j}{name}', 
                                    ini_val(rand),
                                     0, 
                                     up, 
                                     step) 
                      if  fixed_coef!=i else 1 for i in range(j+1)]
        else:
            coefsR = [zfit.Parameter(f'{name}c^{i}_{j}{name}', 
                                        ini_val(rand),low, up, step,)
                    if  fixed_coef!=i else 1 for i in range(j+1)]
        if truncated: models[j] = truncated_bernstein(coefsR, obs)
        else: models[j] = bernstein(coefsR, obs)
    return models
    
    
def check_params_at_zero(minimum, threshold = 1e-3):
    params_at_zero = list()
    free_params = list()
    at_zero = False
    for p_obj, p_res in minimum.params.items():
        if p_obj.value().numpy() < threshold:
            at_zero = True
            params_at_zero.append(p_obj)
        else:
            free_params.append(p_obj)
    return  at_zero, free_params, params_at_zero




def create_zfit_loss(data, models, obs, weights='none'):
    data  = np.array(data)
    if np.all(weights=='none'):
        dataZ = zfit.Data.from_numpy(obs=obs, array= data)
    else:
        dataZ = zfit.Data.from_numpy(obs=obs, array= data,
                                     weights = np.array(weights))
    return dataZ

        
        
def minimize_models(data, models, obs, hesse=True, return_nlls=False, weights='none'):
    
    dataZ = create_zfit_loss(data, models, obs, weights)
    minimums = dict()
    nlls = dict()
    
    for deg, model in models.items():
        #print(deg)
        nlls[deg] = zfit.loss.UnbinnedNLL(model,dataZ)
        #any_param_floating = any(list([p.floating for p in model.params.values()]))
        #if any_param_floating:
        minuit = zfit.minimize.Minuit()
        try:
            minimize = minuit.minimize(nlls[deg])
        except RuntimeError:
            print('Minimization failed continuing...')
            zfit.util.cache.clear_graph_cache()
            minimums[deg] = None
            continue
        print(minimize)
        if hesse:
            try:
                minimize.hesse() 
            except Exception as e: print(e)
        minimums[deg] = minimize
        zfit.util.cache.clear_graph_cache()
        # else:
        #     minimums[deg] = None
    if return_nlls:
        return minimums, nlls
    else:
        return minimums

    
def minimize_models_refit(data, models, obs, hesse=True, return_nlls=False, weights='none'):
    """Make NLL minimizations of polynomials. The `models` polynomials must be values of a dictionary whose index is the degree.
       Data is a list, array, or pd.Series? and if data is weighted, weights must be an array of the same size as `data`
       Returns dictionary of minimums indexes as in `models`
    """
    minuit = zfit.minimize.Minuit()
    dataZ = create_zfit_loss(data, models, obs, weights)
    minimums = dict()
    nlls = dict()
    
    for deg, model in models.items():
        minimums[deg] = list()
        nlls[deg] = zfit.loss.UnbinnedNLL(model,dataZ)

        if any(list([p.floating for p in model.params.values()])):
            #First Fit
            minimize = minuit.minimize(nlls[deg])
            if hesse:
                minimize.hesse() 
            minimums[deg].append(minimize)
            
            #Check if there is any paramter that must be zero. Threshold is at 1e-3
            at_zero, free_params, params_at_zero = check_params_at_zero(minimize)
            
            #Second fit if any param at zero
            if at_zero:
                for p in params_at_zero: p.set_value(0)
                minimize = minuit.minimize(nlls[deg], params=free_params)
                if hesse:
                    minimize.hesse() 
                minimums[deg].append(minimize)
            
            zfit.util.cache.clear_graph_cache()
        else:
            minimums[deg].append(None)
        
    if return_nlls:
        return minimums, nlls
    else:
        return minimums
    

        

def evaluate_best_chi2(data, models, minimums, nbins=20, strategy='close1', weights='none', display=True, out_dir=None):
    """Evaluates the chi2 of each model given binned data with nbins uniform bins.
     Meant to be used after `minimize_models_refit`
     It also selects the best model given an `strategy`
        
        strategy:
            close1: Get the best model with chi2/dof = 1
            min   : Get the best model with minimum chi2/dof

        dof :: Number of bins - Number of free parameters
    """
    chi2_list = list()
    chi2_dof = list()

    # if not display:
    #     plt.ioff()
    for deg, model in models.items():

        if type(model)==list():
            model=model[-1]
        figure = plt.figure(figsize=(10,12))
        #figure.suptitle('Efficien', fontsize=25, y=0.93)
        _1 = plt.subplot2grid(shape=(100,1), loc=(0,0), rowspan=76, fig = figure)
        _2 = plt.subplot2grid(shape=(100,1), loc=(78,0), rowspan=20, fig = figure)

        h, chi2 = plot_tools.plot_model(data=data, weights=weights,
                               pdf=model, axis=_1, bins=nbins,  
                               return_chi2=True, pulls=True, 
                               axis_pulls=_2 , chi_x=0.5, chi_y=0.2)
        
        #Choose the last minimum if refit
        if type(minimums[deg])==list:
            dof = nbins-len(model.params)
        else:
            dof = nbins-len(model.params)
        chi2_list.append(chi2)
        chi2_dof.append(chi2/dof)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f'Deg_{deg}.png'), 
                        bbox_inches='tight')
        if display:
            plt.show()
        plt.close()

    chi2_list = np.array(chi2_list)
    chi2_dof = np.array(chi2_dof)
    if strategy=='close1':
        best_deg = np.argmin(np.abs(chi2_dof-1))
    elif strategy=='minimum':
        best_deg = np.argmin(chi2_dof)
    else:
        raise NotImplementedError
    
    plt.plot(range(len(chi2_dof)), chi2_dof)
    plt.scatter(range(len(chi2_dof)), chi2_dof)
    plt.scatter(best_deg, chi2_dof[best_deg], marker='*', color='red')
    plt.ylabel('chi^2 / dof')
    plt.xlabel('degree')
    plt.xticks(range(deg))
    if out_dir:
        plt.savefig(os.path.join(out_dir, f'Summary.png'), bbox_inches='tight')
    if display:
        plt.show()
    plt.close()

    return models[best_deg], best_deg, chi2_list, chi2_dof


"""def init_params_c(model, c=0.1):
    for param in model.get_params():
        param.set_value(c)
        
        
def find_param(model, name):
    for par in model.get_params(): 
        if par.name==name: return par"""


        
def get_best_chi2_model(data, models, obs, nbins, name, 
                        refit=False, return_new_model=True,
                        weights = 'none', return_minimum=False,
                        family='bernstein', display=True, out_dir=None,
                        return_chi2s = False,
                        strategy='close1',):
    #Initialize parameters
    for deg, model in models.items():
        tools.init_params_c(model, family)


    if refit:
        minimums  =  minimize_models_refit(data, models, obs, weights=weights)
    else:
        minimums  =  minimize_models(data, models, obs, weights=weights)

    best_model, best_deg,  chi2_list, chi2_dof = evaluate_best_chi2(data, models, minimums,
                                              nbins=20, 
                                              weights=weights, 
                                              display=display,
                                              strategy=strategy,
                                              out_dir=out_dir)
    print(best_deg)

    best_deg_minimum = minimums[best_deg]
    if type(best_deg_minimum)==list: 
        best_deg_minimum = best_deg_minimum[-1]
    
    if not return_new_model:
        if return_minimum:
            return best_model, best_deg_minimum, minimums
        return best_model
    
    if family=='bernstein':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 0 
            if best_par in best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        new_model = bernstein(coefsR, obs, name=name)
        
    
    elif family=='non_neg_cheby':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 0 
            if best_par in best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        new_model = non_negative_chebyshev(coefsR, obs, name=name)
        
        
        
    elif family=='chebyshev':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 1
            if best_par in  best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        new_model = zfit.pdf.Chebyshev(obs, coefsR[1:], coeff0=coefsR[0], name=name)

    else:
        raise NotImplementedError
    
    if return_minimum:
            if return_chi2s :
                return new_model, best_deg_minimum, minimums, chi2_list, chi2_dof
            return new_model, best_deg_minimum, minimums
    return new_model







def get_best_Ftest_model_(data, models, obs, nbins, name, 
                        refit=False, return_new_model=True,
                        weights = 'none', return_minimum=False,
                        family='bernstein', display=True, out_dir=None,
                        return_chi2s = False,
                        strategy='close1',):
    
    #Initialize parameters
    for deg, model in models.items():
        tools.init_params_c(model, family)


    if refit:
        minimums  =  minimize_models_refit(data, models, obs, weights=weights)
    else:
        minimums  =  minimize_models(data, models, obs, weights=weights)

    

    # best_model, best_deg,  chi2_list, chi2_dof = evaluate_best_chi2(data, models, minimums,
    #                                           nbins=20, 
    #                                           weights=weights, 
    #                                           display=display,
    #                                           strategy=strategy,
    #                                           out_dir=out_dir)
    best_model, best_deg,  chi2_list, chi2_dof = evaluate_best_Ftest(data, models, minimums,
                                              nbins=nbins, 
                                              weights=weights, 
                                              display=display,
                                              strategy=strategy,
                                              out_dir=out_dir)
    print(best_deg)

    best_deg_minimum = minimums[best_deg]
    if type(best_deg_minimum)==list: 
        best_deg_minimum = best_deg_minimum[-1]
    
    if not return_new_model:
        if return_minimum:
            return best_model, best_deg_minimum, minimums
        return best_model
    if family=='bernstein':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 0 
            if best_par in best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        new_model = bernstein(coefsR, obs, name=name)
        
    
    elif family=='non_neg_cheby':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 0 
            if best_par in best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        new_model = non_negative_chebyshev(coefsR, obs, name=name)
        
        
        
    elif family=='chebyshev':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 1
            if best_par in  best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        new_model = zfit.pdf.Chebyshev(obs, coefsR[1:], coeff0=coefsR[0], name=name)

    else:
        raise NotImplementedError
    
    if return_minimum:
            if return_chi2s :
                return new_model, best_deg_minimum, minimums, chi2_list, chi2_dof
            return new_model, best_deg_minimum, minimums
    return new_model


def get_best_Ftest_model(data, models, obs, nbins, name, 
                        refit=False, return_new_model=True,
                        weights = 'none', return_minimum=False,
                        family='bernstein', display=True, out_dir=None,
                        return_chi2s = False,
                        strategy='close1',
                        type_='phsp', Bin=''):
    #Initialize parameters
    for deg, model in models.items():
        tools.init_params_c(model, family)


    if refit:
        minimums  =  minimize_models_refit(data, models, obs, weights=weights)
    else:
        minimums  =  minimize_models(data, models, obs, weights=weights)


    best_model, best_deg,  chi2_list, chi2_dof = evaluate_best_Ftest(data, models, minimums,
                                              nbins=nbins, 
                                              weights=weights, 
                                              display=display,
                                              strategy=strategy,
                                              out_dir=out_dir, 
                                              type_=type_,
                                              Bin=Bin)

    best_deg_minimum = minimums[best_deg]
    if type(best_deg_minimum)==list: 
        best_deg_minimum = best_deg_minimum[-1]
    
    if not return_new_model:
        if return_minimum:
            return best_model, best_deg_minimum, minimums
        return best_model
    
    if family=='bernstein':
        coefsR = list()
        for i in range(best_deg+1):
            name_ = f'c^{i}_{best_deg}'
            best_par = tools.find_param_substring(best_model, name_)
            best_val = 0 
            if best_par in best_deg_minimum.params:
                best_val = best_deg_minimum.params[best_par]['value']
                
            coefsR.append(zfit.Parameter(f'{name}_{name_}', 
                                         best_val, 
                                         #0, 10, 0.0001
                                        ) )
        
        new_model_ = bernstein(coefsR, obs, name=name)

        
        if 'gauss' in best_model.name.lower():
            print('SUMPDF!!!!')
            
            mu_p_par    = tools.find_param_substring(best_model, 'mu+')
            mu_m_par    = tools.find_param_substring(best_model, 'mu-')
            sigma_p_par = tools.find_param_substring(best_model, 'sigma+')
            sigma_m_par = tools.find_param_substring(best_model, 'sigma-')
            frac_par    = tools.find_param_substring(best_model, 'frac')
            
            mu_p    = zfit.Parameter(f'{name}_mu+', best_deg_minimum.params[mu_p_par]['value'])
            mu_m    = zfit.Parameter(f'{name}_mu-', best_deg_minimum.params[mu_m_par]['value'])
            sigma_p = zfit.Parameter(f'{name}_sigma+', best_deg_minimum.params[sigma_p_par]['value'])
            sigma_m = zfit.Parameter(f'{name}_sigma-', best_deg_minimum.params[sigma_m_par]['value'])
            gauss1 = zfit.pdf.Gauss(mu_p, sigma_p, obs)
            gauss2 = zfit.pdf.Gauss(mu_m, sigma_m, obs)
            double_gauss = zfit.pdf.SumPDF([gauss1, gauss2], [0.5], obs, name='Sum of Gaussians')

            #frac_   = zfit.Parameter(f'{name}_frac', best_deg_minimum.params[frac_par]['value'])
            if not frac_par:
                new_model = double_gauss

            else:
                # mu_p    = zfit.Parameter(f'{name}_mu+', best_deg_minimum.params[mu_p_par]['value'])
                # mu_m    = zfit.Parameter(f'{name}_mu-', best_deg_minimum.params[mu_m_par]['value'])
                # sigma_p = zfit.Parameter(f'{name}_sigma+', best_deg_minimum.params[sigma_p_par]['value'])
                # sigma_m = zfit.Parameter(f'{name}_sigma-', best_deg_minimum.params[sigma_m_par]['value'])
                frac_   = zfit.Parameter(f'{name}_frac', best_deg_minimum.params[frac_par]['value'])
                # gauss1 = zfit.pdf.Gauss(mu_p, sigma_p, obs)
                # gauss2 = zfit.pdf.Gauss(mu_m, sigma_m, obs)
                new_model=zfit.pdf.SumPDF([double_gauss,new_model_], [frac_], obs)
        else:
            new_model=new_model_




    
    if return_minimum:
            if return_chi2s :
                return new_model, best_deg_minimum, minimums, chi2_list, chi2_dof
            return new_model, best_deg_minimum, minimums
    return new_model

    

    

def evaluate_best_Ftest(data, models, minimums, nbins=20, strategy='close1', weights='none', display=True, out_dir=None, type_='phsp', Bin='', pval_chi2_min=0.05):
    """Evaluates the chi2 of each model given binned data with nbins uniform bins.
     Meant to be used after `minimize_models_refit`
     It also selects the best model given an `strategy`
        
        strategy:
            close1: Get the best model with chi2/dof = 1
            min   : Get the best model with minimum chi2/dof

        dof :: Number of bins - Number of free parameters
    """
    names_types  =dict(left='Left SB', right='Right SB', phsp='Efficiency') 


    print('MODELS::: ', models)
    chi2_list = list()
    chi2_dof  = list() 
    p_Vals_chi= dict()
    for new_model, key,  minimum in zip(models.values(),minimums.keys(), minimums.values()):

        fig = plt.figure(figsize=[10,10])
        axes = plot_tools.create_axes_for_pulls(fig)
        
        data_h,chi2,exp_h = plot_tools.plot_model(data, new_model, axis=axes[0], bins=nbins, 
                              plot_components=False, pulls=True, chi_x=0.05, chi_y=0.9,
                              axis_pulls=axes[1],
                              print_params=minimum,
                              params_text_opts={'fontsize': 12, 'ncol':2, 'x':[0.35,0.65], 'y':0.03,
                                                'bbox':dict(facecolor='white', alpha=0.5, boxstyle='round'),
                                                'verticalalignment':'bottom', 
                                                'horizontalalignment':'center', 
                                                'zorder':1000,
                                               }, 
                              remove_string='test_nominal',
                              weights=weights, 
                              return_expected_evts=True,
                              integrate=True
                             )
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f'Deg_{key}.png'), 
                        bbox_inches='tight')
        if display:
            plt.show()
        plt.close()

        chi2_list.append(chi2)
        if minimum:
            dof = nbins-len(minimum.params)
        else:
            dof = nbins - 1
        chi2_dof.append(chi2/dof)
        chi2_list.append(chi2)
        p_Vals_chi[key] = 1-stats.chi2.cdf(float(chi2), dof)

    chi2_list = np.array(chi2_list)
    chi2_dof  = np.array(chi2_dof)
    




    #Find out if any model passed the minimum requirement
    #print('Min  : ', pval_chi2_min)
    p_Vals_chi_vals = np.array(list(p_Vals_chi.values())) 
    if not any(p_Vals_chi_vals>pval_chi2_min):
        pval_chi2_min = np.max(p_Vals_chi_vals)*0.99
    #print('Min2 : ', pval_chi2_min)
    
    #### Produce RSS (residual sum squared) for each model vs the data
    RSS = dict()
    n_params = dict()
    first_good_model = False
    # Setting a dummy condition if all p values are zero
    # Since this does not allow to evaluate the RSS for any model!
    if np.all(p_Vals_chi_vals==0):
        first_good_model=True
    for deg, model in models.items():
        print(f'deg {deg} : ', p_Vals_chi[deg])
        if not first_good_model and p_Vals_chi[deg] <= pval_chi2_min: 
            continue
        first_good_model=True
        n_params[deg] = len(model.get_params())
        model_histogram = plot_tools.bin_model(model, bins=data_h[1], 
                                                 verbose=False, 
                                                 integrate=False)*np.sum(data_h[0])
        RSS[deg] = np.sum((data_h[0]-model_histogram)**2)    

    print(RSS)






    # Get the best degree in the recursive way
    best_degrees = []
    p_vals_tmp = []
    best_degrees_index = [0]
    best_deg_tmp = []
    comparison = []
    textos = ['F-test (p value)']
    KEYS = list(RSS.keys())
    KEYS.sort()
    print(KEYS)
    for enum, key1 in enumerate(KEYS):
        
        if enum==0: best_degrees.append(key1)
        n_param1 = n_params[key1]
        rss1 = RSS[key1]
        if len(best_degrees_index)>1 and key1>best_degrees[-1]: continue

        for key2 in KEYS:
            rss2 = RSS[key2]
            n_param2 = n_params[key2]
            if n_param1>=n_param2: continue
            if n_param1<=best_degrees[-1]: continue
            st, pv = get_f_test(rss1, n_param1, rss2, n_param2, nbins)
            p_vals_tmp.append(pv)
            comparison.append(f'{key1} vs {key2}')
            best_deg_tmp.append(key2)
            textos.append(f'{key1} vs {key2} =  {round(pv,4)}')
            if pv<=0.05:
                best_degrees_index.append(len(p_vals_tmp)-1)
                best_degrees.append(key2)
                break
    
    #In this case, no comparison met the 0.05 threshold, so we take the one with the lowest pval. If the lowest p val is greater than 0.1 we take the initial model
    print(best_degrees, p_vals_tmp)
    if len(p_vals_tmp)==0:
        p_vals_tmp.append(0.5)
    if len(best_degrees)==1 and np.min(p_vals_tmp)<0.1 :
        index_best_deg = np.argmin(p_vals_tmp)
        best_deg = int(comparison[index_best_deg].split('vs')[1].strip())
    elif len(best_degrees)==0:
        import pdb
        pdb.set_trace()
    else:
        index_best_deg = best_degrees_index[-1]
        best_deg = best_degrees[-1]




    fig, ax = plt.subplots()


    if len(textos)<30:
        ax.text(1.03, 0.98, '\n'.join(textos),
                  fontsize=12, va='top', ha='left',
                  transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
    elif len(textos)<90:
        half_1 = textos[:int(len(textos)/2)+1]
        half_2 = textos[int(len(textos)/2)+1:]
        ax.text(1.03, 0.98, '\n'.join(half_1),
                  fontsize=12, va='top', ha='left',
                  transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
        ax.text(1.23, 0.98, '\n'.join(half_2),
                  fontsize=12, va='top', ha='left',
                  transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
    else:
        third_1 = textos[:int(len(textos)/3)+1]
        third_2 = textos[int(len(textos)/3)+1:int(len(textos)*2/3)+1]
        third_3 = textos[int(len(textos)*2/3)+1:]
        ax.text(1.03, 0.98, '\n'.join(third_1),
                  fontsize=12, va='top', ha='left',
                  transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
        ax.text(1.23, 0.98, '\n'.join(third_2),
                  fontsize=12, va='top', ha='left',
                  transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
        ax.text(1.43, 0.98, '\n'.join(third_3),
                  fontsize=12, va='top', ha='left',
                  transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
    
    max_pvals = 14
    min_indx = np.clip(index_best_deg-int(max_pvals/2), a_min=0, a_max=None)
    if len(p_vals_tmp)<14:
        p_vals_to_plot = p_vals_tmp
        comparisons_to_plot = comparison
    else:
        p_vals_to_plot = p_vals_tmp[min_indx:min_indx+max_pvals]
        comparisons_to_plot = comparison[min_indx:min_indx+max_pvals]

    ax.plot(range(len(p_vals_to_plot)), p_vals_to_plot,  color='indianred', ls=':', alpha=0.6)
    ax.scatter(range(len(p_vals_to_plot)), p_vals_to_plot, marker='s', color='indianred', label='F-test')
    ax.set_xticks(range(len(p_vals_to_plot)))
    ax.set_xticklabels(comparisons_to_plot, rotation=90)
    ax.tick_params( axis='x',  which='minor', bottom=False,top=False, )

   
    ax.set_ylabel('F-test  (p value)')
    ax.axhline(0.05, ls='-.', color='grey', label='0.05 threshold')

    indx_best_degree_05=-1
    indx_best_degree_10=-1
    for indx_bd, val_ in enumerate(p_vals_to_plot):
        if val_<0.05: indx_best_degree_05=indx_bd
        if val_<0.1: indx_best_degree_10=indx_bd

    if indx_best_degree_05>=0:
        indx_best_degree = indx_best_degree_05
    elif indx_best_degree_10>=0:
        indx_best_degree = np.argmin(p_vals_to_plot)
    else:
        indx_best_degree=-1

    ax.axvline(indx_best_degree , 
                      label='Selected Degree', 
                      color='blue', 
                      ls='--', linewidth=3 )


    ax.set_xlabel('Degree of Bernstein Polynomial')
    ax.set_ylim(-0.02, 1.02)

    if out_dir:
        plt.savefig(os.path.join(out_dir, f'Summary.png'), bbox_inches='tight')
    if display:
        plt.show()
    plt.close()









    fig, axes = plt.subplots(1,4, figsize=[45,10])
    colors=['crimson', 'green', 'cyan', 'lime', 'magenta', 'orange', 
            'dodgerblue', 'teal', 'sienna', 'deeppink', 'gold',
            'orangered', 'peru', 'slateblue', 'royalblue', 'darkviolet' ]
    ls=['--', '-.', ':', '--', '-.', ':', 
        (0, (1,1)), (0, (5,1)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)),
        (0, (1,1)), (0, (5,1)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))]    
    
    if len(textos)<30:
        axes[-1].text(1.03, 0.98, '\n'.join(textos),
                  fontsize=12, va='top', ha='left',
                  transform=axes[-1].transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
    elif len(textos)<90:
        half_1 = textos[:int(len(textos)/2)+1]
        half_2 = textos[int(len(textos)/2)+1:]
        axes[-1].text(1.03, 0.98, '\n'.join(half_1),
                  fontsize=12, va='top', ha='left',
                  transform=axes[-1].transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
        axes[-1].text(1.23, 0.98, '\n'.join(half_2),
                  fontsize=12, va='top', ha='left',
                  transform=axes[-1].transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
    else:
        third_1 = textos[:int(len(textos)/3)+1]
        third_2 = textos[int(len(textos)/3)+1:int(len(textos)*2/3)+1]
        third_3 = textos[int(len(textos)*2/3)+1:]
        axes[-1].text(1.03, 0.98, '\n'.join(third_1),
                  fontsize=12, va='top', ha='left',
                  transform=axes[-1].transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
        axes[-1].text(1.23, 0.98, '\n'.join(third_2),
                  fontsize=12, va='top', ha='left',
                  transform=axes[-1].transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
        axes[-1].text(1.43, 0.98, '\n'.join(third_3),
                  fontsize=12, va='top', ha='left',
                  transform=axes[-1].transAxes, bbox={'facecolor': 'grey', 'alpha': 0.1, 'boxstyle':"round"})
    

    axes[-1].plot(range(len(p_vals_to_plot)), p_vals_to_plot,  color='indianred', ls=':', alpha=0.6)
    axes[-1].scatter(range(len(p_vals_to_plot)), p_vals_to_plot, marker='s', color='indianred', label='F-test')
    axes[-1].set_xticks(range(len(p_vals_to_plot)))
    axes[-1].set_xticklabels(comparisons_to_plot, rotation=90)
    axes[-1].tick_params( axis='x',  which='minor', bottom=False,top=False, )
    axes[-1].set_ylabel('F-test  (p value)')
    axes[-1].axhline(0.05, ls='-.', color='grey', label='0.05 threshold')


    axes[-1].axvline(indx_best_degree , 
                      label='Selected Degree', 
                      color='blue', 
                      ls='--', linewidth=3 )


    axes[-1].set_xlabel('Degree of Bernstein Polynomial')
    axes[-1].set_ylim(-0.02, 1.02)




    keys_models   = list(models.keys()) 
    keys_models.sort()
    models_per_ax = int(len(models)/3)
    best_deg_ = best_deg
    if -1 in keys_models:
        best_deg_ = best_deg+1
    for indx_ax in range(3):
        ax = axes[indx_ax]
        main_ks = dict(color=colors[0], linewidth=2, ls=ls[0])

        if best_deg_==models_per_ax*indx_ax:
            main_ks = dict(color='blue', 
                           linewidth=3)
       
        if indx_ax ==2:
            keys_ax = keys_models[indx_ax*models_per_ax:]
        else:
            keys_ax = keys_models[indx_ax*models_per_ax:(indx_ax+1)*models_per_ax]
        
        _plot_ini = plot_tools.plot_model(data, 
                                      models[keys_ax[0]], 
                                      axis=ax, 
                                      weights=weights, 
                                      bins=nbins, 
                                      main_kwargs=main_ks, 
                                      pdf_name=models[keys_ax[0]].name+f'  ({str(round(p_Vals_chi[keys_ax[0]],3))})',
                                      return_chi2=True)
        for indx, key_model in enumerate(keys_ax[1:]):

            main_ks = dict(color=colors[indx+1], linewidth=2, ls=ls[indx+1])
            if best_deg_==models_per_ax*indx_ax+indx+1:
                main_ks = dict(color='blue', 
                               linewidth=4)
            plot_tools.model(models[key_model], scaling=_plot_ini[0], axis=ax, 
                             label=models[key_model].name+f'  ({str(round(p_Vals_chi[key_model],3))})',
                             **main_ks)
        ax.legend(frameon=True, ncol=2, fontsize=13)
    axes[1].set_title(f'$q^2$ Bin {Bin} - {names_types[type_.lower()]}')
    legend_1 = axes[-1].legend(frameon=True, #loc='upper right',
                        title=f'{names_types[type_.lower()]}\n$q^2$Bin {Bin}', fontsize=13, title_fontsize=15)
    if out_dir:
        plt.savefig(os.path.join(out_dir+'../../',f"{names_types[type_.lower()].replace(' ','')}_Bin{Bin}.pdf"),
               bbox_inches='tight')
        #plt.savefig(os.path.join(out_dir, f'Summary.png'), bbox_inches='tight')
    if display:
        plt.show()
    plt.close()


    return models[best_deg], best_deg, chi2_list, chi2_dof


    
    
    
    
    
def main2():
    
    import time
    start = time.time()
    nbins = 20
    Bin = sys.argv[1]
    bins = sys.argv[2]
    positive=''
    matched= ''
    if len(sys.argv)>2: positive = sys.argv[2]
    if len(sys.argv)>3: matched = sys.argv[3]
    

    if Bin != 'Complete': Bin = int(Bin)
    
    path = tools.analysis_path('DataSelection/MonteCarlo/PHSP_Signal_Feb2021/Skim2/Skim4/joined-3Abr_0/')
    print(path)

    #path = '/home/horacio/Documents/hcl/DataSelection/MonteCarlo/PHSP_Signal_Feb2021/Skim2/Skim4/joined-3/'

    feb21 = {'Complete':pd.read_pickle(path+'Complete.pkl')}
    for index in range(-1,12):
        feb21[index] = pd.read_pickle(path+f'Bin_{index}.pkl')
        #mass_mask = (feb21[index].BMass>=5.0) & (feb21[index].BMass<=5.7) & feb21[index].GENCand
        #feb21[index] = feb21[index][mass_mask]
    
    dataframe = feb21
    nbins = nbins
    
    
    
    
    
    #CREATE MODELS
    cos = zfit.Space('cos', limits=[-1,1])
    models = dict()
    for j in range(0, nbins-1):
        if positive:
            coefsR = [zfit.Parameter(f'c_{i}_{j}', 0.1, 0, 10, 0.001)  for i in range(j+1)]
        else:
            coefsR = [zfit.Parameter(f'c_{i}_{j}', value=0.1,)  for i in range(j+1)]
        models[j] = bernstein(coefsR, cos)
    
    
    
    #####----------->>>> ----------->>>> ----------->>>> ----------->>>> ----------->>>> 
    #####----------->>>> ----------->>>> ----------->>>> ----------->>>> ----------->>>> 
    if matched:
        print('Previous: ', len(dataframe[Bin]))
        mass_mask = (dataframe[Bin].BMass>=5.0) & (dataframe[Bin].BMass<=5.7) & dataframe[Bin].GENCand
        dataframe[Bin] = dataframe[Bin][mass_mask]
        print('After   : ', len(dataframe[Bin]))
    #####----------->>>> ----------->>>> ----------->>>> ----------->>>> ----------->>>> 
    #####----------->>>> ----------->>>> ----------->>>> ----------->>>> ----------->>>> 
    
    
    
    #CREATE MINIMIZE
    data_array = np.array(dataframe[Bin].cosThetaKMu)
    data = zfit.Data.from_numpy(obs=cos, array= data_array)
    chi2DOF = list()
    mimimums = dict()
    for deg, complete in models.items():
        nll = zfit.loss.UnbinnedNLL(complete,data)
        minuit = zfit.minimize.Minuit()
        minimize = minuit.minimize(nll)
        mimimums[deg] = minimize
        
        
        
        
    #PLOT AND EVALUATE CHI2
    chi2_list = list()
    chi2_dof = list()
    name_output = f'Bernstein_Bin{Bin}_DELETEME_integrate'
    
    os.makedirs('../plots/', exist_ok=True) 
    pdf = PdfPages(f'../plots/{name_output}.pdf')
    for deg, model in models.items():

        figure = plt.figure(figsize=(10,12))
        #figure.suptitle('Efficien', fontsize=25, y=0.93)
        _1 = plt.subplot2grid(shape=(100,1), loc=(0,0), rowspan=76, fig = figure)
        _2 = plt.subplot2grid(shape=(100,1), loc=(78,0), rowspan=20, fig = figure)

        h, chi2 = plot_tools.plot_model(data=dataframe[Bin].cosThetaKMu, 
                       pdf=model, 
                       axis=_1, 
                       bins=nbins,  
                       return_chi2 = True, 
                       pulls=True, 
                       axis_pulls=_2)
        chi2_list.append(chi2)
        chi2_dof.append(chi2/(nbins-len(model.params)+1))
        pdf.savefig()
        
    
    
    
    
    #DEFINE THE BEST POLYNOMIAL
    min_deg = range(0,len(chi2_dof))[np.argmin(chi2_dof)]
    min_chi2dof = chi2_dof[np.argmin(chi2_dof)]

    
    
    
    
    #SAVE INFORMATION OF THE BEST POLYNOMIAL
    best_model_parameters=dict()
    for p, err in mimimums[min_deg].hesse().items():
        best_model_parameters[p.name] = {'val':p.value().numpy(), 'err':err['error']}
        
    texto = f'MIN CHI2/DOF {round(min_chi2dof, 3)}     BERNSTEIN POL {min_deg}\n'
    for order, result in best_model_parameters.items():
        texto+= f'$C_{order.split("_")[1]}$ = {round(result["val"],3)}$\pm${round(result["err"],3)}\n'
        
    plt.figure()
    plt.title(f'{Bin} Bin')
    x, y = (len(chi2_dof)+1)/2, max(chi2_dof)*0.8
    plt.text(x,y, texto, fontsize=15, ha='center', va='center',
             bbox=dict(facecolor='none', edgecolor='grey', color='grey', alpha=0.01))
    plt.scatter(range(0,len(chi2_dof)), chi2_dof)
    plt.plot(range(0,len(chi2_dof)), chi2_dof)
    plt.xticks(range(0,len(chi2_dof),2))
    plt.ylabel('$\chi^2/DOF$')
    plt.xlabel('Bernstein Polynomial Degree')
    pdf.savefig()
    
    
    
    plt.figure()
    plt.title(f'{Bin} Bin   Log Scale')
    x, y = (len(chi2_dof)+1)/2, max(chi2_dof)*0.7
    plt.text(x,y, texto, fontsize=15, ha='center', va='center',
             bbox=dict(facecolor='none', edgecolor='grey', color='grey', alpha=0.01))
    plt.scatter(range(0,len(chi2_dof)), chi2_dof)
    plt.plot(range(0,len(chi2_dof)), chi2_dof)
    plt.xticks(range(0,len(chi2_dof),2))
    plt.ylabel('$\chi^2/DOF$')
    plt.xlabel('Bernstein Polynomial Degree')
    plt.yscale('log')
    pdf.savefig()
    
    
    pdf.close()
    
    
    os.makedirs('../jsons/', exist_ok=True) 
    with open(f'../jsons/{name_output}.json', 'w+') as jj:
        json.dump(best_model_parameters, jj, indent=4)

    print('\n\n\n')
    print(time.time()-start)




def main(debug = True):
    
    cos = zfit.Space('cos', limits=[-1,1])
    if debug:
        import pdb
        pdb.set_trace()
    
    
    
    #Create Bernstein models
    models = create_bernstein_models(cos, 10, positive_constraint=True, rand=False)
    
    #Create some data toy
    gaussians = [zfit.pdf.Gauss(i, abs(i)/2 +0.001, cos) 
                 for i in np.linspace(-1,1,6)]
    fracs  = [1/len(gaussians) for gaussian in range(len(gaussians)-1)]
    sum_gauss = zfit.pdf.SumPDF(gaussians, fracs=fracs, obs=cos)
    sampler = sum_gauss.create_sampler(5000)
    sampler.resample()
    data    = sampler.numpy()
    
    minimums  =  BernsteinEffy.minimize_models(data, models, cos)
    

    if debug: pdb.set_trace()    
    
    
    
    
if __name__ == '__main__':
    #main()

    x = zfit.Space('x', [-2,2])
    bernstein_models = create_bernstein_models(x, 20, positive_constraint=True)

    data_ = np.random.normal(0, 0.5, 500)
    #data_ = np.random.uniform(-2, 2, 500)
    new_model, best_deg_minimum, minimums, chi2_list, chi2_dof = get_best_chi2_model(data_, bernstein_models, x, 10, 'Best', 
                        refit=True, 
                        return_new_model=True,
                        weights = 'none', 
                        return_minimum=True,
                        return_chi2s = True,
                        family='bernstein', 
                        display=False, 
                        out_dir=None,
                        strategy='close1',)



    print('BEST MODEL : ', new_model)
    print(best_deg_minimum)

    print('\n\n')
    for deg,chi2,chi2_dof in zip(range(len(bernstein_models)),chi2_list,chi2_dof ):
        print(deg, '\t', chi2, '\t', chi2_dof )

