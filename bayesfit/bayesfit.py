from __future__ import division, print_function

import numpy as np
import scipy.optimize as so

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
