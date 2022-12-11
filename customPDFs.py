import tensorflow as tf
from tensorflow.math import erfc, erf, atan, exp
import numpy as np
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import math
import zfit
import scipy

version =  zfit.__version__.split('.')
if int(version[1])>=5:
    from zfit import z
else:
    from zfit import ztf as z

from zfit.models.dist_tfp import WrapDistribution
from collections import OrderedDict 
from scipy.special import binom
#from scipy import special
from scipy.integrate import quad
import re
import json
import tools

class exGaussian(zfit.pdf.BasePDF):
    """Positive Exponential Distribution convoluted with Gaussian
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution"""
    def __init__(self, lambda_, sigma, obs, name="exGaussian" ):
        # se debe definir los parametros a pasar a la pdf
        params = {
              'lambda_': lambda_,
              'sigma': sigma, }
        super().__init__(obs, params, name=name)

    def _unnormalized_pdf(self, x):
        x_unstacked = z.unstack_x(x)
        lambda_, sigma = self.params['lambda_'], self.params['sigma']
        sigma_2 = sigma*sigma
        a = (lambda_/2)*tf.exp( (lambda_/2)*(sigma_2*lambda_-2*x_unstacked))
        b = 1-erf((sigma_2*lambda_-x_unstacked)/(sigma*np.sqrt(2.0)))
        return a*b


class decayWidth(zfit.pdf.BasePDF):  
    """
    Decay width of the decay:
    bu -> su mu mu
    https://arxiv.org/pdf/0709.4174.pdf
    https://arxiv.org/pdf/hep-ex/0604007.pdf
    """
    def __init__(self, AFB, FH, obs, name="angular_dist" ):
        # se debe definir los parametros a pasar a la pdf
        params = {
              'AFB': AFB,
              'FH': FH, }
        super().__init__(obs, params, name=name)


    def _unnormalized_pdf(self, x):
        cos_l = z.unstack_x(x)

        AFB, FH = self.params['AFB'], self.params['FH']

        # aqui definimos la pdf
        cos2_l = cos_l*cos_l

        pdf = 3/4*(1-FH)*(1-cos2_l)
        pdf += 1/2*FH
        pdf += AFB*cos_l

        return pdf

    def cdf(self, x):
        """First naive implementation of th cdf using the inherited integration"""

        #Extract the values to evaluate the cdf
        cos_l = z.unstack_x(x)
        #Extract the limits of integration
        limits = self.norm_range.limit1d

        # Since quad takes floats, we cannot pass it a list or array
        # Therefore we iterate over each value
        # Takes too much time, how can we improve it?
        cdfs = list()   
        for val in cos_l:
            # The output of quad is a tuple, where the first entry is the value and the second the error
            #integral = quad(self.pdf, limits[0], val)[0]
            if val<=limits[0]: integral=0
            else:              integral = self.integrate([[limits[0]], [val]]).numpy()[0]
            cdfs.append(integral)
        #Convert it to a np array
        cdfs = np.array(cdfs)
        #Extract the normalization
        norm = self.integrate([[limits[0]], [limits[1]]]).numpy()[0]
        return cdfs/norm
        
    
    
class non_negative_chebyshev(zfit.pdf.BasePDF):
    """
    A wrapper to the numpy chebyshev polynomials, but restraining to be non-negative
    When the polynomial becomes negative we truncate it to zero
    """
    def __init__(self, coeffs, obs, name="Chevyshev" ):        
        self.degree = len(coeffs)-1
        params = dict()
        for indx,c in enumerate(coeffs):
            params[f'c{indx}'] = c

        super().__init__(obs, params, name=name+f' Deg. {self.degree}')


    def _unnormalized_pdf(self, x):
        x_ = z.unstack_x(x)
        limits = self.norm_range.limit1d
        deg = self.degree
        coeffs = [self.params[f'c{i}'] for i in range(deg+1)]
        #cheby = np.polynomial.chebyshev.Chebyshev(coeffs, limits)
        #un_normpdf = cheby(x_)
        #un_normpdf = np.clip(un_normpdf, 0, None)
        cheby_pdf = zfit.pdf.Chebyshev(self.obs, coeffs[1:], coeff0=coeffs[0])
        cheby = cheby_pdf.unnormalized_pdf(x_)
        cheby = tf.where(cheby<0, tf.zeros_like(cheby), cheby)
        return cheby

    
    
    
class bernstein(zfit.pdf.BasePDF):  
    """
    Bernstein_nth Degree
    From a to b
    x-> (x-a/b-a)
    https://en.wikipedia.org/wiki/Bernstein_polynomial
    """
    def __init__(self, coeffs, obs, name="Bernstein" ):        
        self.degree = len(coeffs)-1
        params = dict()
        for indx,c in enumerate(coeffs):
            params[f'c{indx}'] = c

        super().__init__(obs, params, name=name+f' Deg. {self.degree}')


    def _unnormalized_pdf(self, x):
        x_ = z.unstack_x(x)
        limits = self.norm_range.limit1d
        x_T  = (x_-limits[0])/(limits[1]-limits[0])
        deg = self.degree

        basis = dict()
        for i in range(deg+1):
            basis[i] = self.params[f'c{i}']*binom(deg,i)*tf.pow(x_T,i)*tf.pow(1-x_T,deg-i)

        pdf = basis[0]
        for i in range(1, deg+1):
            pdf += basis[i]

        return pdf



    def cdf1(self, x):
        """First naive implementation of th cdf using the inherited integration"""

        #Extract the values to evaluate the cdf
        cos_l = z.unstack_x(x)
        #Extract the limits of integration
        limits = self.norm_range.limit1d

        # Since quad takes floats, we cannot pass it a list or array
        # Therefore we iterate over each value
        # Takes too much time, how can we improve it?
        cdfs = list()   
        for val in cos_l:
            # The output of quad is a tuple, where the first entry is the value and the second the error
            #integral = quad(self.pdf, limits[0], val)[0]
            if val<=limits[0]: integral=0
            else:              integral = self.integrate([[limits[0]], [val]]).numpy()[0]
            cdfs.append(integral)
        #Convert it to a np array
        cdfs = np.array(cdfs)
        #Extract the normalization
        norm = self.integrate([[limits[0]], [limits[1]]]).numpy()[0]
        return cdfs/norm


    def cdf(self, x):
        """Eq. 2.5 from: https://doi.org/10.1016/j.aml.2010.11.013 """
        x_ = z.unstack_x(x)
        x_ = tf.sort(x_)
        limits = self.norm_range.limit1d
        x_T  = (x_-limits[0])/(limits[1]-limits[0])
        deg = self.degree
        factor = (limits[1]-limits[0])/(deg+1)

        basis = dict()
        for i in range(deg+1):
            
            #Obtaining the components of the integration for each basis with the formula in the reference
            basis_2 = dict()
            for j in range(i+1, deg+2):
                basis_2[j]= factor*binom(deg+1,j)*tf.pow(x_T,j)*tf.pow(1-x_T,deg+1-j)
            basis[i] = basis_2[i+1]

            #For each integrated basis sum all its components
            for j in range(i+2, deg+2):
                basis[i] += basis_2[j]
            basis[i] *= self.params[f'c{i}']

        # Sum all components of the pdf
        cdf = basis[0]
        for i in range(1, deg+1):
            cdf += basis[i]

        # Since these forumlas are not normalized Bernstein polynomials, we need to normalize it
        normalization = self.normalization([[limits[0]], [limits[1]]])
		# It seems that the scipy and scikit-gof tools does not handle well tensors, 
        # so we need to cast it to a numpy array 
        cdf_normalized = cdf/normalization
        return cdf_normalized.numpy()

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

def read_single_berntsein_polynomial(obs, params, name='', fixed_params=True, previous_name=''):

    if type(params)==str:
        with open(params, 'r') as jj: params = json.load(jj)
        
    coefs_names = [k for k in params.keys() if 'c^' in k]
    if previous_name:
        coefs_names = [k for k in coefs_names if previous_name in k]
    deg, ordered_coefs = get_degree(coefs_names)
    if not fixed_params:
        coefs = [zfit.Parameter(f'{name}c^{i}_{deg}', params[c]['value']) for i, c in enumerate(ordered_coefs)]
        return bernstein(coefs, obs, name)
    else:
        coefs = [ params[c]['value'] for c in ordered_coefs]
        return bernstein(coefs, obs, name)


# class bernstein_double_gauss(zfit.pdf.BasePDF):  
#     """
#     Bernstein_nth Degree
#     From a to b
#     x-> (x-a/b-a)
#     https://en.wikipedia.org/wiki/Bernstein_polynomial
#     """
#     def __init__(self, mu_minus, sigma_minus, mu_plus, sigma_plus, coeffs, obs, name="DoubleGauss+Bernstein" ):        
        
#         params = dict()

#         self.degree = len(coeffs)-1
#         params['mu_minus'] = mu_minus
#         params['mu_plus'] = mu_plus
#         params['sigma_minus'] = sigma_minus
#         params['sigma_plus'] = sigma_plus

#         for indx,c in enumerate(coeffs):
#             params[f'c{indx}'] = c

#         super().__init__(obs, params, name=name+f' Deg. {self.degree}')


#     def _unnormalized_pdf(self, x):
#         x_ = z.unstack_x(x)
#         limits = self.norm_range.limit1d
#         x_T  = (x_-limits[0])/(limits[1]-limits[0])
#         deg = self.degree

#         basis = dict()
#         for i in range(deg+1):
#             basis[i] = self.params[f'c{i}']*special.binom(deg,i)*tf.pow(x_T,i)*tf.pow(1-x_T,deg-i)

#         pdf = basis[0]
#         for i in range(1, deg+1):
#             pdf += basis[i]

#         pdf += tf.exp(-0.5*tf.pow((x_T - self.params['mu_plus'])/self.params['sigma_plus']  ,2))/(self.params['sigma_plus']*np.sqrt(2*np.pi))
#         pdf += tf.exp(-0.5*tf.pow((x_T - self.params['mu_minus'])/self.params['sigma_minus'],2))/(self.params['sigma_minus']*np.sqrt(2*np.pi))

#         return pdf

def read_2gauss_berntsein_polynomial(obs, params, name='', fixed_params=True, previous_name=''):

    if type(params)==str:
        with open(params, 'r') as jj: params = json.load(jj)
        
    params_keys = [k for k in params.keys() if previous_name in k]

    coefs_names = [k for k in params_keys if 'c^' in k] 
    mu_minus_name    = [k for k in params_keys if 'mu-' in k][0]
    mu_plus_name     = [k for k in params_keys if 'mu+' in k][0]
    sigma_minus_name = [k for k in params_keys if 'sigma-' in k][0]
    sigma_plus_name  = [k for k in params_keys if 'sigma+' in k][0]

    if coefs_names:
        frac_name   = [k for k in params_keys if 'frac' in k][0]
        deg, ordered_coefs = get_degree(coefs_names)

    if not fixed_params:
        
        mu_minus = zfit.Parameter(f'{name}mu-', params[mu_minus_name]['value'],-1, 0, 0.0001)
        sigma_minus = zfit.Parameter(f'{name}sigma-',  
                                     params[sigma_minus_name]['value'], 
                                    0.0001, 2*params[sigma_minus_name]['value'], 0.0001 )
        gauss_minus = zfit.pdf.Gauss(mu_minus, sigma_minus, obs, )#name=name)

        mu_plus  = zfit.Parameter(f'{name}mu+', params[mu_plus_name]['value'] , 0, 1, 0.0001)
        sigma_plus  = zfit.Parameter(f'{name}sigma+', 
                                     params[sigma_plus_name]['value'], 
                                     0.0001, 2*params[sigma_plus_name]['value'], 0.0001)
        gauss_plus  = zfit.pdf.Gauss(mu_plus , sigma_plus, obs, )#name=name)

        double_gauss = zfit.pdf.SumPDF([gauss_minus, gauss_plus], [0.5], obs, name=f'{name}DoubleGuass')

        if coefs_names:
            coefs = [zfit.Parameter(f'{name}c^{i}_{deg}', params[c]['value']) for i, c in enumerate(ordered_coefs)]
            bernstein_ = bernstein(coefs, obs, name)
            frac_ = zfit.Parameter(f'{name}frac', params[frac_name]['value'],0, 1, 0.0001)
            sum_model = zfit.pdf.SumPDF([double_gauss,bernstein_], [frac_], obs, name=f'{name}DoubleGuass+Bernstein')
            return sum_model

        else:
            return double_gauss
    else:

        mu_minus = params[mu_minus_name]['value']
        sigma_minus = params[sigma_minus_name]['value']                                   
        gauss_minus = zfit.pdf.Gauss(mu_minus, sigma_minus, obs, name=name+'Gauss_minus')

        mu_plus  = params[mu_plus_name]['value']
        sigma_plus  = params[sigma_plus_name]['value']
        gauss_plus  = zfit.pdf.Gauss(mu_plus , sigma_plus, obs, name=name+'Gauss_plus')

        double_gauss = zfit.pdf.SumPDF([gauss_minus, gauss_plus], [0.5], obs, name=f'{name}DoubleGuass')

        if coefs_names:
            coefs = [ params[c]['value'] for c in ordered_coefs]
            bernstein_ = bernstein(coefs, obs, name)
            frac_ = params[frac_name]['value']
            sum_model = zfit.pdf.SumPDF([double_gauss,bernstein_], [frac_], obs, name=f'{name}DoubleGuass+Bernstein')
            return sum_model

        else:
            return double_gauss




class truncated_bernstein(zfit.pdf.BasePDF):  
    """
    Bernstein_nth Degree
    From a to b
    x-> (x-a/b-a)
    https://en.wikipedia.org/wiki/Bernstein_polynomial
    """
    def __init__(self, coeffs, obs, name="Bernstein" ):        
        self.degree = len(coeffs)-1
        params = dict()
        for indx,c in enumerate(coeffs):
            params[f'c{indx}'] = c

        super().__init__(obs, params, name=name+f' Deg. {self.degree}')


    def _unnormalized_pdf(self, x):
        x_ = z.unstack_x(x)
        limits = self.norm_range.limit1d
        x_T  = (x_-limits[0])/(limits[1]-limits[0])
        deg = self.degree

        basis = dict()
        for i in range(deg+1):
            basis[i] = self.params[f'c{i}']*binom(deg,i)*tf.pow(x_T,i)*tf.pow(1-x_T,deg-i)

        pdf = basis[0]
        for i in range(1, deg+1):
            pdf += basis[i]
        
        pdf = tf.where(pdf<0, tf.zeros_like(pdf), pdf)

        return pdf
    
    

    
    
    
class errf(zfit.pdf.BasePDF):  
    """
    Error Function from scipy evaluated by 1-x
    x -> (x-mu/sigma)
    """
    def __init__(self, mu, sigma, obs, name="ErrorFunction" ):        
        params = {
              'mu': mu,
              'sigma': sigma, }
        super().__init__(obs, params, name=name)



    def _unnormalized_pdf(self, x):
        x_ = z.unstack_x(x)
        #limits = self.norm_range.limit1d
        x_T  = (x_-self.params['mu'])/self.params['sigma']
        pdf_ = erfc(x_T)
        return pdf_ 

    
    
    
    
class atanTF(zfit.pdf.BasePDF):  
    """
    Arctan fucntion
    x -> (x-mu/sigma)
    """
    def __init__(self, mu, sigma, obs, name="ArcTan" ):        
        params = {
              'mu': mu,
              'sigma': sigma, }
        super().__init__(obs, params, name=name)

    def _unnormalized_pdf(self, x):
        print(x_)
        #x_T  = (x_-self.params['mu'])/self.params['sigma']
        pdf_ = atan((self.params['mu']-x_)/self.params['sigma'])
        return pdf_ # erf(-x) = -erf(x)

    
    
class JohnsonSU(zfit.models.dist_tfp.WrapDistribution):
    """
    Johnson's S_U distribution callback from tensorflowprobability
    https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JohnsonSU
    """
    _N_OBS = 1
    def __init__(self, gamma, delta, mu, sigma, obs, name="JohnsonSU" ):
        gamma, delta, mu, sigma = self._check_input_params(gamma, delta, mu, sigma)
        params = OrderedDict((('gamma', gamma), ('delta', delta), ('mu', mu), ('sigma', sigma)))
        dist_params = lambda: dict(skewness=gamma.value(), tailweight=delta.value(), loc=mu.value(), scale=sigma.value())
        distribution = tfp.distributions.JohnsonSU
        super().__init__(distribution=distribution, dist_params=dist_params, obs=obs, params=params, name=name)

    def mean(self):
        gamma = self.params['gamma'].value()
        delta = self.params['delta'].value()
        mu    = self.params['mu'].value()
        sigma = self.params['sigma'].value()
        return mu-sigma*np.exp(-1/(2*delta**2)*np.sinh(gamma/delta))
        
    def median(self):
        gamma = self.params['gamma'].value()
        delta = self.params['delta'].value()
        mu    = self.params['mu'].value()
        sigma = self.params['sigma'].value()
        return mu+sigma*np.sinh(-gamma/delta)
        
        
class gauss2D(zfit.pdf.BasePDF):  

    def __init__(self, mu1, mu2, s1, s2, rho, obs, name="2Dgauss" ):
        # se debe definir los parametros a pasar a la pdf
        params = {
              'mu1': mu1,
              'mu2': mu2, 
              's1' : s1,
              's2' : s2,
              'rho': rho}
        super().__init__(obs, params, name=name)


    def _unnormalized_pdf(self, x):
        #cos_l, mass = x.unstack_x()
        x1, x2 = zfit.ztf.unstack_x(x)

        mu1, mu2 = self.params['mu1'], self.params['mu2']
        #mu = [mu1, mu2]
        
        s1, s2 = self.params['s1'], self.params['s2']
        rho = self.params['rho']
        coef = -1/(2*(1-rho*rho))
        coef_x  = (x1-mu1)*(x1-mu1)/(s1*s1)
        coef_y  = (x2-mu2)*(x2-mu2)/(s2*s2)
        coef_xy = -2*rho*(x1-mu1)*(x2-mu2)/(s2*s2)
        #cov = [[s1, s12],
        #       [s12, s2]]
        #_pdf = tfd.MultivariateNormalFullCovariance(
        #            loc=mu,
        #            covariance_matrix=cov)

        return exp(coef*(coef_x+coef_y+coef_xy))





#### ! #### ! #### ! #### ! #### ! #### ! #### ! #### ! ####
####   ####  To make the 2D Signal + Background Model   ####
#### ! #### ! #### ! #### ! #### ! #### ! #### ! #### ! ####

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
    return JohnsonSU(gamma, delta, mu, sigma, obs, name=f'JohnsonSU_SignalMass{name}')


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




def read_complete_model_2components_V8(mass, 
    cos, 
    params, 
    name='', 
    fixed_params=True,
    afb_ini='none', 
    fh_ini='none',
    mass_red='none',
    return_2d_Background=False,
    return_angular_signal=False,
    return_efficiency=False,
    return_1D_models=False
    ):

    if type(params)==str:
        params_dict = tools.read_json(params)
    else:
        params_dict = params

    #Mass Models:
    john    = read_johnson(mass,   params_dict, fixed_params=fixed_params, name=name)
    gausexp = read_gauss_exp(mass, params_dict, fixed_params=fixed_params, name=name)
    if mass_red:
        john.set_norm_range(mass_red)
        gausexp.set_norm_range(mass_red)
        gausexp.pdfs[0].set_norm_range(mass_red)
        gausexp.pdfs[1].set_norm_range(mass_red)
    #Angular Models
    leftSB     = read_single_berntsein_polynomial(cos, params_dict, previous_name='Left', fixed_params=fixed_params, name='Left'+name)
    efficiency = read_single_berntsein_polynomial(cos, params_dict, previous_name='Eff', fixed_params=fixed_params, name='Efficiency'+name)

    if any('mu' in p  and 'Right' in p for p in params_dict.keys()):
        rightSB = read_2gauss_berntsein_polynomial(cos, 
                                    params_dict, 
                                    previous_name='Right',
                                    name = f'Right{name}', fixed_params=False) 
    else:
        rightSB = read_single_berntsein_polynomial(cos, 
                                            params_dict, 
                                            previous_name='Right',
                                            name = f'Right{name}', fixed_params=False) 

    #AngularSignalModel
    if afb_ini=='none': AFB = zfit.Parameter('AFB'+name, params_dict['AFB']['value'])
    else: AFB = zfit.Parameter('AFB'+name, afb_ini)
    if fh_ini=='none': FH  = zfit.Parameter('FH'+name, params_dict['FH']['value'])
    else: FH  = zfit.Parameter('FH'+name, fh_ini)

    Decay_rate = decayWidth(AFB, FH, cos, name='DecayRate'+name)
    Decay_rate_eff = zfit.pdf.ProductPDF([Decay_rate,efficiency], obs=cos, name=r'Decay$\times$Eff'+name)
    
    #AngularBackgroundModel
    frac_names = ['frac_SB', 'fracSB', '$fracSB$']
    for frac_name in frac_names:
        if frac_name in params_dict: break
    try:
        #frac= zfit.Parameter('$frac_SB$'+name,params_dict['frac_SB']['value'])
        frac= zfit.Parameter('$frac_SB$'+name,params_dict[frac_name]['value'])
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
    johnson_extended = john.create_extended(Ys)
    gausexp_extended = gausexp.create_extended(Yb)
    MassProjec  = zfit.pdf.SumPDF([johnson_extended, 
                                   gausexp_extended], name='MassProj.'+name )

    projections = dict(mass=MassProjec, angular=AngularProjec)
    if return_2d_Background:
        projections['background']=BackgroundModel

    if return_angular_signal:
        projections['angular_signal']=Decay_rate_eff

    if return_efficiency:
        projections['efficiency'] = efficiency

    if return_1D_models:
        projections['efficiency']      = efficiency
        projections['mass_signal']     = johnson_extended
        projections['mass_background'] = gausexp_extended
        projections['rightSB'] = rightSB        
        projections['leftSB']  = leftSB

    return CompleteModel, projections
