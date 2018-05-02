from __future__ import division, print_function

import numpy as np
import scipy.optimize as so

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
        raise TypeError('Initial optimization failed: ' + res.message)

    #Optimize using BFGS
    res = so.minimize(negL,res.x,args,'BFGS')
    if not res.success:
        raise TypeError('Optimization failed: ' + res.message)

    return res.x, res.hess_inv
