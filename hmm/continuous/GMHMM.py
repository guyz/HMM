'''
Created on Nov 12, 2012

@author: GuyZ
'''

from _ContinuousHMM import _ContinuousHMM
import numpy

class GMHMM(_ContinuousHMM):
    '''
    A Gaussian Mixtures HMM - This is a representation of a continuous HMM,
    containing a mixture of gaussians in each hidden state.
    
    For more information, refer to _ContinuousHMM.
    '''

    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        See _ContinuousHMM constructor for more information
        '''
        _ContinuousHMM.__init__(self,n,m,d,A,means,covars,w,pi,min_std,init_type,precision,verbose) #@UndefinedVariable
        
    def _pdf(self,x,mean,covar):
        '''
        Gaussian PDF function
        '''        
        covar_det = numpy.linalg.det(covar);
        
        c = (1 / ( (2.0*numpy.pi)**(float(self.d/2.0)) * (covar_det)**(0.5)))
        pdfval = c * numpy.exp(-0.5 * numpy.dot( numpy.dot((x-mean),covar.I), (x-mean)) )
        return pdfval