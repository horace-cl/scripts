import tools  
import numpy as np
import pandas as pd
import index_tools
import os
import json
import zfit 
import customPDFs
import re
# import mass_models
import BernsteinEffy
import pdb

path_complete = tools.analysis_path('/CompleteFit/')
path_data     = tools.analysis_path('/CompleteFit/NoteV5/')


def read_johnson(obs, params, name='', fixed_params=True):
    gamma = params['$\\gamma$']['value']
    delta = params['$\\delta$']['value']
    mu    = params['$\\mu$']['value']
    sigma = params['$\\sigma$']['value']
    if not fixed_params:
        gamma = zfit.Parameter('$\\gamma$'+name, gamma)
        delta = zfit.Parameter('$\\delta$'+name, delta)
        mu    = zfit.Parameter('$\\mu$'+name   , mu)
        sigma = zfit.Parameter('$\\sigma$'+name, sigma)
    
    return customPDFs.JohnsonSU(gamma, delta, mu, sigma, obs, name=f'JohnsonSU_SignalMass{name}')



def read_gauss_exp(obs, params, name='', fixed_params=True):
    mu     = params['$\\mu_B$']['value']
    sigma  = params['$\\sigma_B$']['value']
    lambda_= params['$\\lambda_B$']['value']
    frac  = params['frac_mass']['value']
    if not fixed_params:
        mu     = zfit.Parameter('$\\mu_B$'+name, mu)
        sigma  = zfit.Parameter('$\\sigma_B$'+name, sigma)
        lambda_= zfit.Parameter('$\\lambda_B$'+name, lambda_)
        frac  = zfit.Parameter('frac_mass'+name, frac, 0, 1)    
    gauss  = zfit.pdf.Gauss(mu = mu, sigma=sigma, obs = obs, name='Gauss_BackMass') 
    exponential = zfit.pdf.Exponential(lambda_=lambda_, obs = obs, name='Exp_BackMass'+name)    
    return zfit.pdf.SumPDF([gauss, exponential], fracs=frac,
                            obs = obs, name = f'Gauss+Exp_BackgroundMass{name}') 


def read_john_gauss_exp(obs, params, name='', fixed_params=True, return_components=True):
    if type(params)==str:
        with open(params, 'r') as jj: params = json.load(jj)
    Ys = params['Ys']['value']
    Yb = params['Yb']['value']
    if not fixed_params:
        Ys = zfit.Parameter('Ys'+name, Ys)
        Yb = zfit.Parameter('Yb'+name, Yb)
    johnson = read_johnson(obs, params, name, fixed_params)
    gausexp = read_gauss_exp(obs, params, name, fixed_params)
    johnson_extended = johnson.create_extended(Ys)
    gausexp_extended = gausexp.create_extended(Yb)
    if return_components:
        return johnson, gausexp, zfit.pdf.SumPDF([johnson_extended, gausexp_extended], name='MassModel'+name)
    return zfit.pdf.SumPDF([johnson_extended, gausexp_extended], name='MassModel'+name)


def get_degree(coefs_names, type_=''):
    search_degree = [re.search(r'_[0-9]+', coef) for coef in coefs_names]
    search_coef_n = [re.search(r'\^[0-9]+_', coef.replace(type_, '')) for coef in coefs_names]
    degrees = [int(match.group(0).replace('_', '')) for match in search_degree]
    coefs   = [int(match.group(0).replace('_', '').replace(r'^', '')) for match in search_coef_n]
    if len(set(coefs))==1 and degrees[0]!=0:
        search_coef_n = [re.search(r'[0-9]+_', coef) for coef in coefs_names]
        coefs   = [int(match.group(1).replace('_', '')) for match in search_coef_n]  
          
    if not len(set(degrees))==1:
        raise NotImplementedError(f'More than one degree in yur coefs!\n{coefs_names}\nCheck them')
    
    names_ = np.array(coefs_names, dtype='O')
    return degrees[0], names_[np.array(coefs).argsort()]


def read_berntsein_polynomial(obs, params, name='', fixed_params=True, type_=''):

    if type(params)==str:
        with open(params, 'r') as jj: params = json.load(jj)

    coefs_names = [k for k in params.keys() if type_.lower() in k.lower()] 
    print(coefs_names)
    deg, ordered_coefs = get_degree(coefs_names,)
    if not fixed_params:
        coefs = [zfit.Parameter(f'{name}c^{i}_{deg}', params[c]['value']) for i,c in enumerate(ordered_coefs)]
        return customPDFs.bernstein(coefs, obs, name)
    else:
        coefs = [ params[c]['value'] for c in ordered_coefs]
        return customPDFs.bernstein(coefs, obs, name)


def read_single_berntsein_polynomial(obs, params, name='', fixed_params=True):

    if type(params)==str:
        with open(params, 'r') as jj: params = json.load(jj)
        
    coefs_names = [k for k in params.keys() if 'c^' in k] 
    deg, ordered_coefs = get_degree(coefs_names)
    if not fixed_params:
        coefs = [zfit.Parameter(f'{name}c^{i}_{deg}', params[c]['value']) for i, c in enumerate(ordered_coefs)]
        return customPDFs.bernstein(coefs, obs, name)
    else:
        coefs = [ params[c]['value'] for c in ordered_coefs]
        return customPDFs.bernstein(coefs, obs, name)
    
    
    
def read_complete_model(mass, cos, params, name='', fixed_params=True, afb_ini='none', fh_ini='none'):

    if type(params)==str:
        params_dict = tools.read_json(params)
    else:
        params_dict = params
    #Mass Models:
    john    = read_johnson(mass,   params_dict, fixed_params=fixed_params, name=name)
    gausexp = read_gauss_exp(mass, params_dict, fixed_params=fixed_params, name=name)
    
    #Angular Models
    leftSB     = read_berntsein_polynomial(cos, params_dict, type_='Left')
    efficiency = read_berntsein_polynomial(cos, params_dict, type_='Eff')
    rightSB    = read_berntsein_polynomial(cos, params_dict, type_='Right')
    
    #AngularSignalModel
    if afb_ini=='none': AFB = zfit.Parameter('AFB'+name, params_dict['AFB']['value'])
    else: AFB = zfit.Parameter('AFB'+name, afb_ini)
    if fh_ini=='none': FH  = zfit.Parameter('FH'+name, params_dict['FH']['value'])
    else: FH  = zfit.Parameter('FH'+name, fh_ini)
    Decay_rate = customPDFs.decayWidth(AFB, FH, cos, name='DecayRate'+name)
    Decay_rate_eff = zfit.pdf.ProductPDF([Decay_rate,efficiency], obs=cos, name=r'Decay$\times$Eff'+name)
    
    #AngularBackgroundModel
    try:
        frac= zfit.Parameter('$frac_SB$'+name,params_dict['frac_SB']['value'])
    except KeyError:
        frac= zfit.Parameter('$fracSB$'+name,params_dict['fracSB']['value'])

    angularBackground = zfit.pdf.SumPDF([leftSB, rightSB], fracs=frac, obs=cos, name=r'AngularBack'+name)
    
    #Signal Yields
    Ys  = zfit.Parameter('signalY'+name,params_dict['Ys']['value'])
    Yb  = zfit.Parameter('backgroundY'+name,params_dict['Yb']['value'])
    
    #Extending Models
    SignalModel    = zfit.pdf.ProductPDF([john, Decay_rate_eff], name='Signal_model'+name).create_extended(Ys)
    BackgroundModel= zfit.pdf.ProductPDF([gausexp, angularBackground], name='Background_model'+name).create_extended(Yb)
    
    CompleteModel  = zfit.pdf.SumPDF([SignalModel, BackgroundModel], name='Complete_model'+name )
    AngularProjec  = zfit.pdf.SumPDF([Decay_rate_eff.create_extended(Ys), 
                                      angularBackground.create_extended(Yb)], name='AngularProj.'+name )
    MassiveProjec  = zfit.pdf.SumPDF([john.create_extended(Ys), 
                                      gausexp.create_extended(Yb)], name='MassProj.'+name )
    
    return CompleteModel, (MassiveProjec, AngularProjec)


def read_complete_model_2components(mass, cos, params, name='', 
                                    fixed_params=True, 
                                    afb_ini='none', fh_ini='none', 
                                    return_2d_Background=False,
                                    return_angular_signal=False, 
                                    return_efficiency=False):

    if type(params)==str:
        params_dict = tools.read_json(params)
    else:
        params_dict = params
    #Mass Models:
    john    = read_johnson(mass,   params_dict, fixed_params=fixed_params, name=name)
    gausexp = read_gauss_exp(mass, params_dict, fixed_params=fixed_params, name=name)
    
    #Angular Models
    leftSB     = read_berntsein_polynomial(cos, params_dict, type_='Left', fixed_params=fixed_params)
    efficiency = read_berntsein_polynomial(cos, params_dict, type_='Eff', fixed_params=fixed_params)
    rightSB    = read_berntsein_polynomial(cos, params_dict, type_='Right', fixed_params=fixed_params)
    
    #AngularSignalModel
    if afb_ini=='none': AFB = zfit.Parameter('AFB'+name, params_dict['AFB']['value'])
    else: AFB = zfit.Parameter('AFB'+name, afb_ini)
    if fh_ini=='none': FH  = zfit.Parameter('FH'+name, params_dict['FH']['value'])
    else: FH  = zfit.Parameter('FH'+name, fh_ini)
    Decay_rate = customPDFs.decayWidth(AFB, FH, cos, name='DecayRate'+name)
    Decay_rate_eff = zfit.pdf.ProductPDF([Decay_rate,efficiency], obs=cos, name=r'Decay$\times$Eff'+name)
    
    #AngularBackgroundModel
    try:
        frac= zfit.Parameter('$frac_SB$'+name,params_dict['frac_SB']['value'])
    except KeyError:
        frac= zfit.Parameter('$fracSB$'+name,params_dict['fracSB']['value'])
        
    angularBackground = zfit.pdf.SumPDF([leftSB, rightSB], fracs=frac, obs=cos, name=r'AngularBack'+name)
    
    #Signal Yields
    Ys  = zfit.Parameter('Ys'+name,params_dict['Ys']['value'])
    Yb  = zfit.Parameter('Yb'+name,params_dict['Yb']['value'])
    
    #Extending Models
    SignalModel    = zfit.pdf.ProductPDF([john, Decay_rate_eff], name='Signal_model'+name).create_extended(Ys)
    BackgroundModel= zfit.pdf.ProductPDF([gausexp, angularBackground], name='Background_model'+name).create_extended(Yb)
    
    CompleteModel  = zfit.pdf.SumPDF([SignalModel, BackgroundModel], name='Complete_model'+name )
    AngularProjec  = zfit.pdf.SumPDF([Decay_rate_eff.create_extended(Ys), 
                                      angularBackground.create_extended(Yb)], name='AngularProj.'+name )
    MassiveProjec  = zfit.pdf.SumPDF([john.create_extended(Ys), 
                                      gausexp.create_extended(Yb)], name='MassProj.'+name )

    projections = dict(mass=MassiveProjec, angular=AngularProjec)
    if return_2d_Background:
        projections['background']=BackgroundModel

    if return_angular_signal:
        projections['angular_signal']=Decay_rate_eff

    if return_efficiency:
        projections['efficiency'] = efficiency

    return CompleteModel, projections



def create_complete_pdf(BIN, mass, cos, version='', name='', afb_ini=0, fh_ini=0.2, path = 'none'):
    if path == 'none':
        cpath  = os.path.join(path_data, f'Bin{BIN}', 'complete_fit.json')
    else:
        cpath = tools.analysis_path(path)
    with open(cpath, 'r') as jj:
        params = json.load(jj)
    return read_complete_model(mass, cos, params, name=name, fixed_params=True, afb_ini=afb_ini, fh_ini=fh_ini)[0]