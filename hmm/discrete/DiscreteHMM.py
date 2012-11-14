'''
Created on Nov 12, 2012

@author: GuyZ
'''

from hmm._BaseHMM import _BaseHMM
import numpy

class DiscreteHMM(_BaseHMM):
    '''
    classdocs
    '''

    def __init__(self,n,m,A=None,B=None,pi=None,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        Constructor
        '''
        _BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable
        
        self.A = A
        self.pi = pi
        self.B = B
        
        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        if init_type == 'uniform':
            self.pi = numpy.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = numpy.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.B = numpy.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)
    
    def _mapB(self,observations):
        self.B_map = numpy.zeros( (self.n,len(observations)), dtype=self.precision)
        
        for j in xrange(self.n):
            for t in xrange(len(observations)):
                self.B_map[j][t] = self.B[j][observations[t]]
                
    def _updatemodel(self,new_model):
        _BaseHMM._updatemodel(self,new_model) #@UndefinedVariable
        
        self.B = new_model['B']
    
    def _reestimate(self,stats,observations):
        # re-estimate A, pi
        new_model = _BaseHMM._reestimate(self,stats,observations) #@UndefinedVariable
        
        # re-estimate the discrete probability of the observable symbols
        B_new = self._reestimateB(observations,stats['gamma'])
        
        new_model['B'] = B_new
        
        return new_model
    
    def _reestimateB(self,observations,gamma):
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