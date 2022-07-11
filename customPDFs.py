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
from scipy.integrate import quad




class decayWidth(zfit.pdf.BasePDF):  
    """
    Decay width of the decay:
    bu -> su mu mu
    https://arxiv.org/pdf/0709.4174.pdf
    https://arxiv.org/pdf/hep-ex/0604007.pdf
    """
    def __init__(self, AFB, FH, obs, name="angular_dist" ):
        # se debe definir los parametros a pasar a la pdf
        params = {'AFB': AFB,'FH': FH, }
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


    # def cdf(self, x):
    #     """First naive implementation of th cdf using the inherited integration"""

    #     #Extract the values to evaluate the cdf
    #     cos_l = z.unstack_x(x)
    #     #Extract the limits of integration
    #     limits = self.norm_range.limit1d

    #     # Since quad takes floats, we cannot pass it a list or array
    #     # Therefore we iterate over each value
    #     # Takes too much time, how can we improve it?
    #     cdfs = list()   
    #     for val in cos_l:
    #         # The output of quad is a tuple, where the first entry is the value and the second the error
    #         #integral = quad(self.pdf, limits[0], val)[0]
    #         if val<=limits[0]: integral=0
    #         else:              integral = self.integrate([[limits[0]], [val]]).numpy()[0]
    #         cdfs.append(integral)
    #     #Convert it to a np array
    #     cdfs = np.array(cdfs)
    #     #Extract the normalization
    #     norm = self.integrate([[limits[0]], [limits[1]]]).numpy()[0]
    #     cdf_norm=cdfs/norm
    #     return cdf_norm.to_numpy()
    

################cdf 
       # def cdf(self,x):
        # min_=self.obs.limits[0]
        # cdf= quad(self.pdf,min_,x)
        # return cdf
        
        # def _cdf_single(self, x, *args):
        # _a, _b = self._get_support(*args)
        # return integrate.quad(self._pdf, _a, x, args=args)[0]

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
        cdf_norm=cdf/normalization
        return cdf_norm.numpy()

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
        pdf_ = atan((x_-self.params['mu'])/self.params['sigma'])
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