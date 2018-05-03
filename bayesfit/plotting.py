from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

def posterior(fitresult,plot_likelihood=False):
    '''
    Plot the posterior probablity/likelihood from the given FitResult object.
    '''

    #The fit parameters and the inverse covariance matrix
    p0 = fitresult.p
    cov = fitresult.cov
    invcov = np.linalg.inv(fitresult.cov)

    #A single parameter
    if p0.size == 1:
        x = np.linspace(p0[0]-4*np.sqrt(cov[0,0]),p0[0]+4*np.sqrt(cov[0,0]),150)
        #L might not be vectorized in terms of p, so will calculate the values
        #separately for compatibility

        L = np.zeros(x.shape)

        for i in range(L.size):
            L[i] = fitresult.L([x[i]])

        #Normalize L
        L = L-np.max(L)

        if plot_likelihood:
            plt.plot(x,L,label='Likelihood')
            plt.plot(x,-0.5*(x-p0[0])**2/cov[0,0],label='Gaussian approximation')
        else:
            #Compute probablity
            p = np.exp(L)

            #Normalize
            p = p/np.trapz(p,x)

            plt.plot(x,p,label='Likelihood')
            plt.plot(x,np.exp(-0.5*(x-p0)**2*invcov[0,0])/np.sqrt(2*np.pi*cov[0,0]),label='Gaussian approximation')

        plt.legend()
