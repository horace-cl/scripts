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
import customPDFs
from random import random




def create_johnson_signal(obs, name=''):
    mu     = zfit.Parameter(r'$\mu$'+f'{name}', 5.279)
    sigma  = zfit.Parameter(r'$\sigma$'+f'{name}', 0.01)
    delta  = zfit.Parameter(r'$\delta$'+f'{name}', 0.01)
    gamma  = zfit.Parameter(r'$\gamma$'+f'{name}', 0.01)
    model = customPDFs.JohnsonSU(gamma = gamma,
                                 delta = delta,
                                 mu    = mu,
                                 sigma = sigma,
                                 obs = obs, 
                                 name = 'mass_signal')
    return model

def create_doubleCB_signal(obs, name=''):
    
    mu    = zfit.Parameter(r'$\mu$'+f'{name}', 5.279)
    sigma = zfit.Parameter(r'$\sigma$'+f'{name}', 0.03)
    
    alphal= zfit.Parameter(r'$\alpha_l$'+f'{name}', 0.5)
    nl    = zfit.Parameter(r'$n_l$'+f'{name}', 2)
    
    alphar= zfit.Parameter(r'$\alpha_r$'+f'{name}', 0.5)
    nr    = zfit.Parameter(r'$n_r$'+f'{name}', 2)
    
    model = zfit.pdf.DoubleCB( mu = mu, sigma = sigma,
                             alphal = alphal, nl = nl,
                             alphar = alphar, nr = nr,
                             obs    = obs )
    
    return model




def create_signal_model(obs, name='', kind='Jhonson'):
    
    if kind=='Jhonson':
        return create_johnson_signal(obs, name=name)
    elif kind=='DoubleCB':
        return create_doubleCB_signal(obs, name=name)
    else:
        raise NotImplementedError


def create_gauss_back(obs, name=''):
    mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', 4.9, 4, 5.1)
    sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', 0.2, 0.05, 1)
    gauss  = zfit.pdf.Gauss(mu = mu, sigma=sigma, obs = obs, name='gauss_mass_back')        
    return gauss

def create_exp_back(obs, name=''):
    lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', -2)
    exponential = zfit.pdf.Exponential(lambda_=lambda_, obs = obs, name='exp_mass_back')
    return exponential

def create_gauss_exp_back(obs, name=''):
    mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', 4.9, 4, 5.1)
    sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', 0.2, 0.005, 1)
    gauss  = zfit.pdf.Gauss(mu = mu, sigma=sigma, obs = obs, name='gauss_mass_back')        
    lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', -2)
    exponential = zfit.pdf.Exponential(lambda_=lambda_, obs = obs, name='exp_mass_back')
    frac = zfit.Parameter(r'frac_mass'+f'{name}', 0.1, 0, 1.0, 0.001)
    
    model = zfit.pdf.SumPDF([gauss, exponential], fracs=frac,
                        obs = obs, name = 'mass_back')
    return model

def create_errf_exp_back(obs, name=''):
    mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', 4.9, 4, 5.1)
    sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', 0.2, 0.05, 1)
    errff  = customPDFs.errf(mu = mu, 
                             sigma=sigma, 
                             obs = obs, 
                             name='gauss_mass_back')        
    lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', -2)
    exponential = zfit.pdf.Exponential(lambda_=lambda_, obs = obs, name='exp_mass_back')
    frac = zfit.Parameter(r'frac_mass'+f'{name}', 0.1, 0, 1.0, 0.001)
    
    model = zfit.pdf.SumPDF([errff, exponential], fracs=frac,
                        obs = obs, name = 'mass_back')
    return model



def create_background_model(obs, name='', kind='GaussExp'):
    
    if kind=='Gauss':
        return create_gauss_back(obs, name=name)
    elif kind=='Exp':
        return create_exp_back(obs, name=name)
    elif kind=='GaussExp':
        return create_gauss_exp_back(obs, name=name)
    elif kind=='ErrfExp':
        return create_errf_exp_back(obs, name=name)
    else:
        raise NotImplementedError
        
        
        
        
        
        
def minimize_model_minuit(data, model, obs, hesse=True):
    data   = np.array(data)
    dataZ  = zfit.Data.from_numpy(obs=obs, array= data)
    minuit = zfit.minimize.Minuit()
    nll    = zfit.loss.UnbinnedNLL(model,dataZ)
    minimize = minuit.minimize(nll)
    zfit.util.cache.clear_graph_cache()
    return minimize


def minimize_extended_model_minuit(data, model, obs, free_params, hesse=True):
    data   = np.array(data)
    dataZ  = zfit.Data.from_numpy(obs=obs, array= data)
    minuit = zfit.minimize.Minuit()
    nll    = zfit.loss.ExtendedUnbinnedNLL(model,dataZ)
    minimize = minuit.minimize(nll, free_params)
    zfit.util.cache.clear_graph_cache()
    return minimize
    



