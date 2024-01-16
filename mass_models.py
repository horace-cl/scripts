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

pretty_names = dict(
    DCB = 'Double CrystallBall',
)

def correlation_to_covariance(correlation_matrix, std_deviations):
    # Calculate the covariance matrix
    cov_matrix = correlation_matrix * np.outer(std_deviations, std_deviations)
    
    return cov_matrix


def create_johnson_signal(obs, name='', mass_mean=[5.279]):
    mu     = zfit.Parameter(r'$\mu_J$'+f'{name}', *mass_mean)
    sigma  = zfit.Parameter(r'$\sigma_J$'+f'{name}', 0.01)
    delta  = zfit.Parameter(r'$\delta_J$'+f'{name}', 0.01)
    gamma  = zfit.Parameter(r'$\gamma_J$'+f'{name}', 0.01)
    model = customPDFs.JohnsonSU(gamma = gamma,
                                 delta = delta,
                                 mu    = mu,
                                 sigma = sigma,
                                 obs = obs, 
                                 name = 'JohnsonSU'+name)
    return model


def create_johnsonGauss_signal(obs, name='', mass_mean=[5.279], same_mu=False):
    mu     = zfit.Parameter(r'$\mu$JG'+f'{name}', *mass_mean)
    sigma  = zfit.Parameter(r'$\sigma$JG'+f'{name}', 0.01)
    delta  = zfit.Parameter(r'$\delta$JG'+f'{name}', 0.01)
    gamma  = zfit.Parameter(r'$\gamma$JG'+f'{name}', 0.01)
    
    JSU = customPDFs.JohnsonSU(gamma = gamma,
                                 delta = delta,
                                 mu    = mu,
                                 sigma = sigma,
                                 obs = obs, 
                                 name = 'JohnsonG'+name)
    
    mu2     = zfit.Parameter(r'$\mu_2$JG'+f'{name}', *mass_mean)
    sigma2  = zfit.Parameter(r'$\sigma_2$JG'+f'{name}', 0.05)
    gauss = zfit.pdf.Gauss(mu if same_mu else mu2, sigma2, obs,
                           name = 'JGauss'+name)
    fraction = zfit.Parameter(r'$frac$_JG'+f'{name}', 0.7, 0, 1)                         
    model = zfit.pdf.SumPDF([JSU, gauss], fracs=[fraction], obs=obs, name='JohnsonGauss'+name)    
                             
    return model

def create_CB_signal(obs, name='', mass_mean=5.279, sigma=0.03, alpha=1, n=2):
    
    mu    = zfit.Parameter(r'$\mu$_CB'+f'{name}', mass_mean)
    if type(sigma)==float:
        sigma = zfit.Parameter(r'$\sigma$_CB'+f'{name}',sigma)
    
    
    alpha= zfit.Parameter(r'$\alpha$_CB'+f'{name}',alpha)
    n    = zfit.Parameter(r'$n$_CB'+f'{name}', n)
    
    model = zfit.pdf.CrystalBall( mu = mu, 
                              sigma = sigma,
                             alpha = alpha, n = n,
                             obs    = obs , 
                              name='CrystalBall'+name)
    
    return model

def create_CBGauss_signal(obs, name='', mass_mean=5.279, sigma=0.03, alpha=1, n=2, same_mu=False):
    
    mu    = zfit.Parameter(r'$\mu$_CBG'+f'{name}', mass_mean)
    sigma = zfit.Parameter(r'$\sigma$_CBG'+f'{name}',sigma)
    sigma_g = zfit.Parameter(r'$\sigma_G$_CBG'+f'{name}',sigma*1.5)

    if same_mu:
        mu_g = mu
    else:
        mu_g = zfit.Parameter(r'$\mu$_CBG_G'+f'{name}', mass_mean)

    
    alpha= zfit.Parameter(r'$\alpha$_CBG'+f'{name}',alpha)
    n    = zfit.Parameter(r'$n$_CBG'+f'{name}', n)
    
    fraction_gauss = zfit.Parameter(r'$frac$_CBG'+f'{name}', 0.01, 0, 0.2)
    
    cb = zfit.pdf.CrystalBall( mu = mu, 
                              sigma = sigma,
                             alpha = alpha, n = n,
                             obs    = obs , )
    g = zfit.pdf.Gauss(mu_g, sigma_g, obs)
    model = zfit.pdf.SumPDF([g, cb], fracs=[fraction_gauss], obs=obs, name='CrystalBallGauss'+name)
    
    return model


def create_doubleCB_signal(obs, name='', mass_mean=[5.279], sigma=[0.03, 0.001, 0.1],
                                alpha_r=[1.2, 0.9, 100], alpha_l=[1.2, 0.9, 100]):
    
    mu    = zfit.Parameter(r'$\mu$_DCB'+f'{name}', *mass_mean)
    if type(sigma)==float:
        sigma = zfit.Parameter(r'$\sigma$_DCB'+f'{name}', sigma, 0.001, 0.1)
    elif type(sigma)==list:
        sigma = zfit.Parameter(r'$\sigma$_DCB'+f'{name}', *sigma)
    
    alphal= zfit.Parameter(r'$\alpha_l$_DCB'+f'{name}',*alpha_l )
    #nl    = zfit.Parameter(r'$n_l$'+f'{name}', 5, 0.08, 300, 0.001)
    #nl    = zfit.Parameter(r'$n_l$_DCB'+f'{name}', 100, 0.08, 300, 0.001)
    nl    = zfit.Parameter(r'$n_l$_DCB'+f'{name}', 100, 1, 300, 0.001)
    
    alphar= zfit.Parameter(r'$\alpha_r$_DCB'+f'{name}', *alpha_r)
    #nr    = zfit.Parameter(r'$n_r$'+f'{name}', 8, 0.08, 300, 0.001)
    #nr    = zfit.Parameter(r'$n_r$_DCB'+f'{name}', 100, 0.08, 300, 0.001)
    nr    = zfit.Parameter(r'$n_r$_DCB'+f'{name}', 100, 1, 300, 0.001)
    
    model = zfit.pdf.DoubleCB( mu = mu, sigma = sigma,
                             alphal = alphal, nl = nl,
                             alphar = alphar, nr = nr,
                             obs    = obs , 
                             name='DoubleCB'+name)
    
    return model


def create_doubleCBGauss_signal(obs, name='', mass_mean=[5.279], 
                                alpha_r=[1.2, 0.9, 100], 
                                alpha_l=[1.2, 0.9, 100],
                                n_l = [100, 0.08, 300, 0.001], 
                                n_r = [100, 0.08, 300, 0.001]):
    
    mu    = zfit.Parameter(r'$\mu$_DCBG'+f'{name}', *mass_mean)
    sigma = zfit.Parameter(r'$\sigma$_DCBG'+f'{name}', 0.03, 0.001, 0.1)
    
    alphal= zfit.Parameter(r'$\alpha_l$_DCBG'+f'{name}',*alpha_l )
    #nl    = zfit.Parameter(r'$n_l$'+f'{name}', 5, 0.08, 300, 0.001)
    nl    = zfit.Parameter(r'$n_l$_DCBG'+f'{name}', *n_l)
    
    alphar= zfit.Parameter(r'$\alpha_r$_DCBG'+f'{name}', *alpha_r)
    #nr    = zfit.Parameter(r'$n_r$'+f'{name}', 8, 0.08, 300, 0.001)
    nr    = zfit.Parameter(r'$n_r$_DCBG'+f'{name}', *n_r)
    
    dcb = zfit.pdf.DoubleCB( mu = mu, sigma = sigma,
                             alphal = alphal, nl = nl,
                             alphar = alphar, nr = nr,
                             obs    = obs, name='DCB_'+name )

    #mu2    = zfit.Parameter(r'$\mu_2$_DCBG'+f'{name}', mass_mean)
    sigmag = zfit.Parameter(r'$\sigma_g$_DCBG'+f'{name}', 0.03, 0.001, 0.1)
    
    gauss = zfit.pdf.Gauss(mu, sigmag, obs, name='Gauss_'+name)
    
    frac = zfit.Parameter(f'frac_DCBG'+name, 0.5, 0, 1)
    
    model = zfit.pdf.SumPDF([dcb, gauss], frac, name='DoubleCBGauss'+name)
    return model


def create_tripleGauss_signal(obs, name='', free_fracs = 0,
                              mass_mean=5.27828, 
                              sigma_1=[0.03],
                              sigma_2=[0.06],
                              sigma_3=[0.07],
                             ):
    
    mu     =  zfit.Parameter(r'$\mu$_TG'+f'{name}',    mass_mean)
    sigma1 = zfit.Parameter(r'$\sigma_1$_TG'+f'{name}', *sigma_1)
    sigma2 = zfit.Parameter(r'$\sigma_2$_TG'+f'{name}', *sigma_2)
    
    mu3    = zfit.Parameter(r'$\mu_3$_TG'+f'{name}',    mass_mean*0.99)
    sigma3 = zfit.Parameter(r'$\sigma_3$_TG'+f'{name}', *sigma_3)
    
    gauss1 = zfit.pdf.Gauss(mu,  sigma1, obs, name='TG_1')
    gauss2 = zfit.pdf.Gauss(mu,  sigma2, obs, name='TG_2')
    gauss3 = zfit.pdf.Gauss(mu3, sigma3, obs, name='TG_3')
    
    
    if free_fracs==0:
        model = zfit.pdf.SumPDF([gauss1, 
                                 gauss2, 
                                 gauss3], 
                                [1/3, 1/3, 1/3], 
                                obs, name=f'TripleGauss{name}')
    elif free_fracs==1:
        frac   = zfit.Parameter('frac'+str(name),   0.87972)

        gauss12= zfit.pdf.SumPDF([gauss1, gauss2], [1/2, 1/2], 
                                 obs)
        model   = zfit.pdf.SumPDF([gauss12, gauss3], 
                                  [frac], 
                                  obs, name=f'TripleGauss{name}')

    
    return model


def create_doubleGauss_signal(obs, name='', mass_mean = 5.27828, window_width=0.7):
    """window_width could be `infer` and it will take the maximum and minimum values from the obs
    however it trows some warning messages to be updated!"""
    max1 = window_width/3
    max2 = window_width/4
    if type(window_width)== str and window_width=='infer':
        window_width = obs.limits[1][0][0]-obs.limits[0][0][0]
    if type(mass_mean)==str:
        mu =  zfit.Parameter(r'$\mu$_2G'+f'{name}',    
                             mass_mean, 
                             mass_mean-max2,
                             mass_mean+max2)
    elif type(mass_mean)==list:
        mu =  zfit.Parameter(r'$\mu$_2G'+f'{name}', *mass_mean)
    
    #sigma1 = zfit.Parameter(r'$\sigma_1$'+f'{name}', 0.01, 0.001, 0.04)
    sigma1 = zfit.Parameter(r'$\sigma_1$_2G'+f'{name}', max1/2, max1/10, max1)
    
    #sigma2 = zfit.Parameter(r'$\sigma_2$'+f'{name}', 0.03, 0.01, 0.07)
    sigma2 = zfit.Parameter(r'$\sigma_2$_2G'+f'{name}', max2/2, max2/10, max2)
    
    gauss1 = zfit.pdf.Gauss(mu,  sigma1, obs)
    gauss2 = zfit.pdf.Gauss(mu,  sigma2, obs)
    
    frac   = zfit.Parameter('frac_2G'+str(name),   0.87972, 0 , 1)

    gauss12= zfit.pdf.SumPDF([gauss1, gauss2], frac, obs, name=f'DoubleGauss{name}')

    
    return gauss12

def create_gauss_signal(obs, name='', mass_mean=5.3):
    mu     =  zfit.Parameter(r'$\mu$_G'+f'{name}',     mass_mean)
    sigma1 = zfit.Parameter(r'$\sigma_1$_G'+f'{name}', 0.02)
    
    gauss1 = zfit.pdf.Gauss(mu,  sigma1, obs, name='Gauss')
    
    return gauss1



def create_signal_model(obs, name='', kind='Jhonson', **kwargs):
    
    if kind.lower()=='jhonson' or kind.lower()=='johnson':
        return create_johnson_signal(obs , name=name, **kwargs)
    elif kind=='DoubleCB' or kind=='DCB':
        return create_doubleCB_signal(obs, name=name, **kwargs)
    elif kind=='GaussDoubleCB' or kind=='GaussDCB':
        return create_doubleCBGauss_signal(obs , name=name, **kwargs)
    elif kind=='GaussCB' or kind=='GaussCB':
        return create_CBGauss_signal(obs , name=name, **kwargs)
    else:
        raise NotImplementedError










def create_gauss_back(obs, name=''):
    mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', 4.9, 4, 5.1)
    sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', 0.2, 0.05, 1)
    gauss  = zfit.pdf.Gauss(mu = mu, sigma=sigma, obs = obs, name='Gaussian')        
    return gauss

def create_exp_back(obs, name=''):
    lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', -2)
    exponential = zfit.pdf.Exponential(lambda_=lambda_, obs = obs, name='Exponential')
    return exponential

def create_errf_back(obs, name=''):
    mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', 5.07, 5, 5.2, 0.0001)
    sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', 0.05, 0.005, 0.1, 0.0001)
    errff  = customPDFs.errf(mu = mu, 
                             sigma=sigma, 
                             obs = obs, 
                             name='errf_mass_back')  
    return errff

def create_gauss_exp_back(obs, name=''):
    mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', 4.9, 3.5, 5.03)
    sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', 0.2, 0.02, 1)
    gauss  = zfit.pdf.Gauss(mu = mu, sigma=sigma, obs = obs, name='Gaussian')        
    lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', -2)
    exponential = zfit.pdf.Exponential(lambda_=lambda_, obs = obs, name='Exponential')
    frac = zfit.Parameter(r'frac_mass'+f'{name}', 0.1, 0, 1.0, 0.001)
    
    model = zfit.pdf.SumPDF([gauss, exponential], fracs=frac,
                        obs = obs, name = 'Gauss+Exp')
    return model

def create_errf_exp_back(obs, name='', 
                         mu_opts=[5.1, 4.5, 5.3],
                         sigma_opts=[0.05, 0.005, 0.1, 0.0001],
                        lambda_opts=[-2],
                        frac_opts= [0.8, 0, 1.0, 0.001], 
                        fixed_params = False):
    if fixed_params:
        mu = mu_opts[0]
        sigma = sigma_opts[0]
        lambda_ = lambda_opts[0]
        frac = frac_opts[0]
    else:
        mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', *mu_opts)
        sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', *sigma_opts)
        lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', *lambda_opts)
        frac = zfit.Parameter(r'frac_mass'+f'{name}', *frac_opts)
        
    #mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', *mu_opts)
    #sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', *sigma_opts)
    errff  = customPDFs.errf(mu = mu, 
                             sigma=sigma, 
                             obs = obs, 
                             name='ErrFunc')        
    #lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}', *lambda_opts)
    exponential = zfit.pdf.Exponential(lambda_=lambda_, 
                                       obs = obs, name='Exponential')
    #frac = zfit.Parameter(r'frac_mass'+f'{name}', *frac_opts)
    
    model = zfit.pdf.SumPDF([errff, exponential], fracs=frac,
                        obs = obs, name = 'Err+Exp')
    return model

def create_atan_exp_back(obs, name='',
                        mu_opts=[5.1, 4.5, 5.3],
                        sigma_opts=[0.005, 0.001, 0.2, 0.0001],
                        lambda_opts=[-2],
                        frac_opts= [0.8, 0, 1.0, 0.001], 
                        fixed_params = False):
    if fixed_params:
        mu = mu_opts[0]
        sigma = sigma_opts[0]
        lambda_ = lambda_opts[0]
        frac = frac_opts[0]
    else:
        mu     = zfit.Parameter(r'$\mu_B$'+f'{name}', *mu_opts)
        sigma  = zfit.Parameter(r'$\sigma_B$'+f'{name}', *sigma_opts)
        lambda_     = zfit.Parameter(r'$\lambda_B$'+f'{name}',
                                     *lambda_opts)
        frac = zfit.Parameter(r'frac_mass'+f'{name}', 
                              *frac_opts)
        
    atan_  = customPDFs.atanTF(mu = mu, 
                             sigma=sigma, 
                             obs = obs, 
                             name='atan_mass_back')        
    exponential = zfit.pdf.Exponential(lambda_=lambda_, 
                                       obs = obs, name='exp_mass_back')    
    model = zfit.pdf.SumPDF([atan_, exponential], fracs=frac,
                        obs = obs, name = 'mass_back')
    return model


def create_background_model(obs, name='', kind='GaussExp', **kwargs):
    
    if kind=='Gauss':
        return create_gauss_back(obs, name=name, **kwargs)
    elif kind=='Exp':
        return create_exp_back(obs, name=name, **kwargs)
    elif kind=='Errf':
        return create_errf_back(obs, name=name, **kwargs)
    elif kind=='GaussExp' or kind=='ExpGauss':
        return create_gauss_exp_back(obs, name=name, **kwargs)
    elif kind.lower()=='errfexp' or kind.lower()=='errexp':
        return create_errf_exp_back(obs, name=name, **kwargs)
    elif kind.lower()=='atanexp':
        return create_atan_exp_back(obs, name=name, **kwargs)
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


def minimize_extended_model_minuit(data, model, obs, free_params, hesse=True, prefit=False):
    if prefit:
        for n,p in enumerate(free_params):
            if 'Y' in p.name: p.set_value(p.value()/10)
        data_pre = np.random.choice(np.array(data), size = int(len(data)/10) ,replace=False )
        dataZ  = zfit.Data.from_numpy(obs=obs, array= data_pre)
        minuit = zfit.minimize.Minuit()
        nll    = zfit.loss.ExtendedUnbinnedNLL(model,dataZ)
        pre_minimize = minuit.minimize(nll, free_params)
        pre_minimize.hesse()

        correlation = pre_minimize.correlation()
        uncertainties  =  np.sqrt(np.diag(pre_minimize.covariance()))
        initial_values = [p.value().numpy() for p in free_params]
        for n,p in enumerate(free_params):
            if 'Y' in p.name:
                uncertainties[n]*=10
                initial_values[n]*=10
                p.set_value(initial_values[n])
                print(f'New values : {initial_values[n]} +- {uncertainties[n]}', )
        initial_covariance = correlation_to_covariance(correlation, uncertainties)
        pre_constraint = zfit.constraint.GaussianConstraint(free_params, 
                                                            initial_values, 
                                                            initial_covariance)

        print('PRE FIT WITH 10% OF DATA')
        print(pre_minimize)
        print('Using minimize result to constraint parameters for the full minimization...\n\n')
        
    else:
        pre_constraint = []                                                        
    data   = np.array(data)
    dataZ  = zfit.Data.from_numpy(obs=obs, array= data)
    minuit = zfit.minimize.Minuit()
    nll    = zfit.loss.ExtendedUnbinnedNLL(model,dataZ, constraints=pre_constraint)
    minimize = minuit.minimize(nll, free_params)
    zfit.util.cache.clear_graph_cache()
    if hesse:
        minimize.hesse()
    return minimize
    



