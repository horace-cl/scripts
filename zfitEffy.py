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
        any_param_floating = any(list([p.floating for p in model.params.values()]))
        if any_param_floating:
            minuit = zfit.minimize.Minuit()
            minimize = minuit.minimize(nlls[deg])
            print(minimize)
            if hesse:
                try:
                    minimize.hesse() 
                except Exception as e: print(e)
            minimums[deg] = minimize
            zfit.util.cache.clear_graph_cache()
        else:
            minimums[deg] = None
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

