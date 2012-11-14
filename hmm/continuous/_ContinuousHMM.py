'''
Created on Nov 12, 2012

@author: GuyZ
'''

from hmm._BaseHMM import _BaseHMM
import numpy

class _ContinuousHMM(_BaseHMM):
    '''
    classdocs
    '''

    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        Constructor
        '''
        _BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable
        
        self.d = d
        self.A = A
        self.pi = pi
        self.means = means
        self.covars = covars
        self.w = w
        self.min_std = min_std

        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        if init_type == 'uniform':
            self.pi = numpy.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = numpy.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.w = numpy.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)            
            self.means = numpy.zeros( (self.n,self.m,self.d), dtype=self.precision)
            self.covars = [[ numpy.matrix(numpy.ones((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
        elif init_type == 'user':
            covars_tmp = [[ numpy.matrix(numpy.ones((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
            for i in xrange(self.n):
                for j in xrange(self.m):
                    if type(self.covars[i][j]) is numpy.ndarray:
                        covars_tmp[i][j] = numpy.matrix(self.covars[i][j])
                    else:
                        covars_tmp[i][j] = self.covars[i][j]
            self.covars = covars_tmp
            
    """
    b[j][Ot] = sum(1...M)w[j][m]*b[j][m][Ot]
    Returns b[j][Ot] based on the current model parameters (means, covars, weights) for the mixtures.
    - j - state
    - Ot - the current observation
    Note: there's no need to get the observation itself as it has been used for calculation before.
    """    
    def _mapB(self,observations):
        self.B_map = numpy.zeros( (self.n,len(observations)), dtype=self.precision)
        self.Bmix_map = numpy.zeros( (self.n,self.m,len(observations)), dtype=self.precision)
        for j in xrange(self.n):
            for t in xrange(len(observations)):
                self.B_map[j][t] = self._calcbjt(j, t, observations[t])
                
    """
    b[j][Ot] = sum(1...M)w[j][m]*b[j][m][Ot]
    Returns b[j][Ot] based on the current model parameters (means, covars, weights) for the mixtures.
    - j - state
    - Ot - the current observation
    Note: there's no need to get the observation itself as it has been used for calculation before.
    """    
    def _calcbjt(self,j,t,Ot):
        bjt = 0
        for m in xrange(self.m):
            self.Bmix_map[j][m][t] = self._pdf(Ot, self.means[j][m], self.covars[j][m])
            bjt += (self.w[j][m]*self.Bmix_map[j][m][t])
        return bjt
    
    """
    Calculates 'gamma_mix' from xi.
    
    Gamma is a (TxNxK) numpy array, where gamma[t][i][m] = the probability of being
    in state 'i' at time 't' with mixture 'm' given the full observation sequence.
    """    
    def _calcgammamix(self,alpha,beta,observations):
        gamma_mix = numpy.zeros((len(observations),self.n,self.m),dtype=self.precision)
        
        for t in xrange(len(observations)):
            for j in xrange(self.n):
                for m in xrange(self.m):
                    alphabeta = 0.0
                    for jj in xrange(self.n):
                        alphabeta += alpha[t][jj]*beta[t][jj]
                    comp1 = (alpha[t][j]*beta[t][j]) / alphabeta
                    
                    bjk_sum = 0.0
                    for k in xrange(self.m):
                        bjk_sum += (self.w[j][k]*self.Bmix_map[j][k][t])
                    comp2 = (self.w[j][m]*self.Bmix_map[j][m][t])/bjk_sum
                    
                    gamma_mix[t][j][m] = comp1*comp2
        
        return gamma_mix
    
    def _updatemodel(self,new_model):
        _BaseHMM._updatemodel(self,new_model) #@UndefinedVariable
        
        self.w = new_model['w']
        self.means = new_model['means']
        self.covars = new_model['covars']
        
    def _calcstats(self,observations):
        stats = _BaseHMM._calcstats(self,observations) #@UndefinedVariable
        stats['gamma_mix'] = self._calcgammamix(stats['alpha'],stats['beta'],observations)

        return stats
    
    def _reestimate(self,stats,observations):
        # re-estimate A, pi
        new_model = _BaseHMM._reestimate(self,stats,observations) #@UndefinedVariable
        
        # re-estimate the continuous probability parameters of the mixtures
        w_new, means_new, covars_new = self._reestimateMixtures(observations,stats['gamma_mix'])
        
        new_model['w'] = w_new
        new_model['means'] = means_new
        new_model['covars'] = covars_new
        
        return new_model
    
    def _reestimateMixtures(self,observations,gamma_mix):
        w_new = numpy.zeros( (self.n,self.m), dtype=self.precision)
        means_new = numpy.zeros( (self.n,self.m,self.d), dtype=self.precision)
        covars_new = [[ numpy.matrix(numpy.zeros((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
        
        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = 0.0
                denom = 0.0                
                for t in xrange(len(observations)):
                    for k in xrange(self.m):
                        denom += (self.eta(t,len(observations)-1)*gamma_mix[t][j][k])
                    numer += (self.eta(t,len(observations)-1)*gamma_mix[t][j][m])
                w_new[j][m] = numer/denom
            w_new[j] = self._normalize(w_new[j])
                
        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = numpy.zeros( (self.d), dtype=self.precision)
                denom = numpy.zeros( (self.d), dtype=self.precision)
                for t in xrange(len(observations)):
                    numer += (self.eta(t,len(observations)-1)*gamma_mix[t][j][m]*observations[t])
                    denom += (self.eta(t,len(observations)-1)*gamma_mix[t][j][m])
                means_new[j][m] = numer/denom
                
        cov_prior = [[ numpy.matrix(self.min_std*numpy.eye((self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = numpy.matrix(numpy.zeros( (self.d,self.d), dtype=self.precision))
                denom = numpy.matrix(numpy.zeros( (self.d,self.d), dtype=self.precision))
                for t in xrange(len(observations)):
                    vector_as_mat = numpy.matrix( (observations[t]-self.means[j][m]), dtype=self.precision )
                    numer += (self.eta(t,len(observations)-1)*gamma_mix[t][j][m]*numpy.dot( vector_as_mat.T, vector_as_mat))
                    denom += (self.eta(t,len(observations)-1)*gamma_mix[t][j][m])
                covars_new[j][m] = numer/denom
                covars_new[j][m] = covars_new[j][m] + cov_prior[j][m]               
        
        return w_new, means_new, covars_new
    
    def _normalize(self, arr):
        summ = numpy.sum(arr)
        for i in xrange(len(arr)):
            arr[i] = (arr[i]/summ)
        return arr
    
    def _pdf(self,x,mean,covar):
        raise NotImplementedError("PDF function must be implemented")
    