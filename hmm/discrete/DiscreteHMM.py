'''
Created on Nov 12, 2012

@author: GuyZ
'''

from hmm._BaseHMM import _BaseHMM
import numpy

class DiscreteHMM(_BaseHMM):
    '''
    A Discrete HMM - The most basic implementation of a Hidden Markov Model,
    where each hidden state uses a discrete probability distribution for
    the physical observations.
    
    Model attributes:
    - n            number of hidden states
    - m            number of observable symbols
    - A            hidden states transition probability matrix ([NxN] numpy array)
    - B            PMFs denoting each state's distribution ([NxM] numpy array)
    - pi           initial state's PMF ([N] numpy array).
    
    Additional attributes:
    - precision    a numpy element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    '''

    def __init__(self,n,m,A=None,B=None,pi=None,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        Construct a new Discrete HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,B,pi), and set the init_type to 'user'.
        
        Normal initialization uses a uniform distribution for all probablities,
        and is not recommended.
        '''
        _BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable
        
        self.A = A
        self.pi = pi
        self.B = B
        
        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        '''
        If required, initalize the model parameters according the selected policy
        '''
        if init_type == 'uniform':
            self.pi = numpy.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = numpy.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.B = numpy.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)
    
    def _mapB(self,observations):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''
        self.B_map = numpy.zeros( (self.n,len(observations)), dtype=self.precision)
        
        for j in xrange(self.n):
            for t in xrange(len(observations)):
                self.B_map[j][t] = self.B[j][observations[t]]
                
    def _updatemodel(self,new_model):
        '''
        Required extension of _updatemodel. Adds 'B', which holds
        the in-state information. Specfically, the different PMFs.
        '''
        _BaseHMM._updatemodel(self,new_model) #@UndefinedVariable
        
        self.B = new_model['B']
    
    def _reestimate(self,stats,observations):
        '''
        Required extension of _reestimate. 
        Adds a re-estimation of the model parameter 'B'.
        '''
        # re-estimate A, pi
        new_model = _BaseHMM._reestimate(self,stats,observations) #@UndefinedVariable
        
        # re-estimate the discrete probability of the observable symbols
        B_new = self._reestimateB(observations,stats['gamma'])
        
        new_model['B'] = B_new
        
        return new_model
    
    def _reestimateB(self,observations,gamma):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the matrix 'B'.
        '''        
        # TBD: determine how to include eta() weighing
        B_new = numpy.zeros( (self.n,self.m) ,dtype=self.precision)
        
        for j in xrange(self.n):
            for k in xrange(self.m):
                numer = 0.0
                denom = 0.0
                for t in xrange(len(observations)):
                    if observations[t] == k:
                        numer += gamma[t][j]
                    denom += gamma[t][j]
                B_new[j][k] = numer/denom
        
        return B_new