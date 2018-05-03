from __future__ import division, print_function

import numpy as np
import scipy.optimize as so

def get_result(output):
    '''
    An auxiliary function to quickly obtain the fit parameters and their
    uncertainties from the fitting function outputs. Basically takes the
    optimized parameters and the square root from the diagonal of the
    covariance matrix and returns vectors p and perr.
    '''

    return output[0], np.sqrt(np.diag(output[1]))

def outlier_fit(f_model,p0,x,y,sigma0,method='conservative'):
    '''
    Least squares fitting algorithm with outlier handling. Fits the given model
    f_model(x,*p) to the (x,y) data. The initial guess for the parameters p0 is
    to be given.

    sigma0 has a slightly different function dependending on the used method.
    With 'conservative' sigma0 is the lower bound for the uncertainty of each
    data point upper value being unlimited. This corresponds to the prior
    sigma0/sigma^2 when sigma >= sigma0, 0 otherwise.

    With 'cauchy' the uncertainties are assumed to be of the same order as
    sigma0 but they can be either smaller or larger. The prior in this case
    is proportional to exp(-sigma0^2/sigma^2)/sigma^2.
    '''

    #define the likelihood for chi squared
    if method=='conservative':
        def L(p):
            R = (f_model(x,*p)-y)/sigma0
            return np.sum(np.log((1-np.exp(-0.5*R**2))/R**2))
    elif method=='cauchy':
        def L(p):
            R = (f_model(x,*p)-y)/sigma0
            return -np.sum(np.log(1+0.5*R**2))

    p, cov = maximize_likelihood(L,p0)
    return p, cov

def least_squares(f_model,p0,x,y,yerr=1,noise_scaling=False):
    '''
    Least squares fitting. Fits the given model f_model(x,*p) to the (x,y) data.
    The initial guess for the parameters p0 is to be given.

    The uncertainties in y yerr can be either a float or an array.
    If error_scaling is False, then the fit is an ordinary least squares.
    If True, then yerr replaced with sigma*yerr, where sigma is treated
    as a nuisance parameter with Jeffreys' prior.
    '''

    #define the likelihood for chi squared
    def L(p):
        return -0.5*np.sum((y-f_model(x,*p))**2/yerr**2)

    p, cov = maximize_likelihood(L,p0)

    if noise_scaling == False:
        return p, cov
    else:
        #Allowing errors to scale corresponds to maximizing L = - N/2*ln(chi^2).
        #This leads to the same optimal as with the ordinary case but with
        #scaled covariance matrix
        return p, -cov*2*L(p)/y.size

def maximize_likelihood(L,x0,args=()):
    '''
    Maximizes the given likelihood function L(x0,*args) starting from
    the initial guess x0. Returns the optimal x with the negative inverse
    of the Hessian of L (i.e. the covariance matrix).
    '''

    #Define the negative of L for minimization
    def negL(x0,*args):
        return -L(x0,*args)

    #Optimize using a robust algorithm
    initial_method = 'Nelder-Mead' #TODO: Could brute force be an option?

    res = so.minimize(negL,x0,args,initial_method)
    if not res.success:
        raise Exception('Initial optimization failed: ' + res.message + '\n' +
                        'Consider adjusting the initial guess.')

    #Optimize using BFGS
    res = so.minimize(negL,res.x,args,'BFGS')
    if not res.success:
        raise Exception('BFGS optimization failed: ' + res.message + '\n' +
                        'Consider adjusting the initial guess.')

    return res.x, res.hess_inv
