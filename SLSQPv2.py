import copy
import inspect
import math
from typing import Callable, Mapping, Optional, Union

from zfit.minimizers.minimizers_scipy import ScipyBaseMinimizerV1
from zfit.minimizers.baseminimizer import (NOT_SUPPORTED, BaseMinimizer, minimize_supports,
                            print_minimization_status)
from zfit.minimizers.termination import CRITERION_NOT_AVAILABLE, ConvergenceCriterion
from zfit.minimizers.strategy import ZfitStrategy
from zfit.minimizers.fitresult import FitResult

class SLSQP(ScipyBaseMinimizerV1):
    def __init__(self,
                 tol: Optional[float] = 1e-8,
                 gradient: Optional[Union[Callable, str]] = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 constraints: Optional[tuple] = (),
                 name: str = "SciPy SLSQP V2",        
                 ) -> None:
        """Local, gradient-based minimizer using tho  Sequential Least Squares Programming algorithm.name.

         `Sequential Least Squares Programming <https://en.wikipedia.org/wiki/Sequential_quadratic_programming>`_
         is an iterative method for nonlinear parameter optimization.

         |@doc:minimizer.scipy.info| This implenemtation wraps the minimizers in
        `SciPy optimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_. |@docend:minimizer.scipy.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            gradient: |@doc:minimizer.scipy.gradient| Define the method to use for the gradient computation
                   that the minimizer should use. This can be the
                   gradient provided by the loss itself or
                   method from the minimizer.
                   In general, using the zfit provided automatic gradient is
                   more precise and needs less computation time for the
                   evaluation compared to a numerical method but it may not always be
                   possible. In this case, zfit switches to a generic, numerical gradient
                   which in general performs worse than if the minimizer has its own
                   numerical gradient.
                   The following are possible choices:

                   If set to ``False`` or ``'zfit'`` (or ``None``; default), the
                   gradient of the loss (usually the automatic gradient) will be used;
                   the minimizer won't use an internal algorithm. |@docend:minimizer.scipy.gradient|
                   |@doc:minimizer.scipy.gradient.internal| ``True`` tells the minimizer to use its default internal
                   gradient estimation. This can be specified more clearly using the
                   arguments ``'2-point'`` and ``'3-point'``, which specify the
                   numerical algorithm the minimizer should use in order to
                   estimate the gradient. |@docend:minimizer.scipy.gradient.internal|

            verbosity: |@doc:minimizer.verbosity| Verbosity of the minimizer. Has to be between 0 and 10.
              The verbosity has the meaning:

               - a value of 0 means quiet and no output
               - above 0 up to 5, information that is good to know but without
                 flooding the user, corresponding to a "INFO" level.
               - A value above 5 starts printing out considerably more and
                 is used more for debugging purposes.
               - Setting the verbosity to 10 will print out every
                 evaluation of the loss function and gradient.

               Some minimizer offer additional output which is also
               distributed as above but may duplicate certain printed values. |@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the `value`, 'gradient` or `hessian`. |@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy| A class of type `ZfitStrategy` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            name: |@doc:minimizer.name| Human readable name of the minimizer. |@docend:minimizer.name|
        """
        options = {}
        minimizer_options = {}
        if options:
            minimizer_options['options'] = options
        if constraints:
            minimizer_options['constraints'] = constraints
            
        scipy_tols = {'ftol': None}

        method = "SLSQP"
        super().__init__(method=method, internal_tol=scipy_tols, gradient=gradient, hessian=NOT_SUPPORTED,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


        
SLSQP._add_derivative_methods(gradient=['2-point', '3-point',
                                               # 'cs',  # works badly
                                               None, True, False, 'zfit'])




def create_constraint(model, afb_index=False, fh_index=False):

    #First look for the indices of the POIs
    if (type(afb_index)!=int and afb_index==False) or (type(fh_index)!=int and fh_index==False):
        #afb_index = False
        #fh_index  = False
        for i,p in enumerate(model.get_params()):
            if 'afb' in p.name.lower() or 'a_fb' in p.name.lower():  afb_index = i
            if 'fh' in p.name.lower() or 'f_h' in p.name.lower():  fh_index = i

        if str(afb_index)=='False' or str(fh_index)=='False':
            print('I was not able to find the indices, please fix it here:\n ../scripts/SLSQP_zfit.py')
            raise NotImplementedError
    #Now define the "simple" constraints give the found index
    constAngParams = (
                 {'type': 'ineq', 'fun': lambda x:  x[fh_index]},
                 {'type': 'ineq', 'fun': lambda x:  3-x[fh_index]},
                 {'type': 'ineq', 'fun': lambda x:  x[fh_index]/2-x[afb_index]},
                 {'type': 'ineq', 'fun': lambda x:  x[fh_index]/2+x[afb_index]}
                )
    print(afb_index, fh_index)

    return constAngParams