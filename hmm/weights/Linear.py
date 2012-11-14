'''
Created on Nov 14, 2012

@author: GuyZ
'''

class Linear(object):
    '''
    Provides the ability to weigh samples in a time series
    as a function time.
    
    This mixin provides an linear weighing function, 
    that can be used to implement time-dependant HMMs.
    '''
    
    def __init__(self):
        self._eta = self._etaf    
    
    def _etaf(self,t,T):
        n = float(t+1)
        m = float(T)
        return n/m   
        