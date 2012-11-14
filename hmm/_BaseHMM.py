'''
Created on Oct 31, 2012

@author: GuyZ

The code is based on hmm.py, from QSTK. See below for license details and information.
'''
import numpy

class _BaseHMM(object):
    '''
    classdocs
    '''
    
    def __init__(self,n,m,precision=numpy.double,verbose=False):
        self.n = n
        self.m = m
        
        self.precision = precision
        self.verbose = verbose
        self.eta = self._eta1
        
    def _eta1(self,t,T):
        return 1.
    
    """
    Forward-Backward procedure is used to efficiently calculate the probability of the observation, given the model - P(O|model)
    alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the observation up to time t, given the model.
    NOTE: only alpha (the forward variable) is actually required to get P(O|model)
    NOTE 2: The returned value is the log-likelihood of the model. It can be greater than one
    when using a continous HMM since pdfs are used, and can provide unnormalized values.
    """   
    def forwardbackward(self,observations):
        alpha = self._calcalpha(observations)
        return numpy.log(sum(alpha[-1]))
    
    """
    Calculates 'alpha' the forward variable.

    The alpha variable is a numpy array indexed by time, then state (TxN).
    alpha[t][i] = the probability of being in state 'i' after observing the 
    first t symbols.
    """
    def _calcalpha(self,observations):
        alpha = numpy.zeros((len(observations),self.n),dtype=self.precision)
        
        # init stage - alpha_1(x) = pi(x)b_x(O1)
        for x in xrange(self.n):
            alpha[0][x] = self.pi[x]*self.B_map[x][0]
        
        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    alpha[t][j] += alpha[t-1][i]*self.A[i][j]
                alpha[t][j] *= self.B_map[j][t]
                
        return alpha

    """
    Calculates 'beta' the backward variable.
    
    The beta variable is a numpy array indexed by time, then state (TxN).
    beta[t][i] = the probability of being in state 'i' and then observing the
    symbols from t+1 to the end (T).
    """
    def _calcbeta(self,observations):
        beta = numpy.zeros((len(observations),self.n),dtype=self.precision)
        
        # init stage
        for s in xrange(self.n):
            beta[len(observations)-1][s] = 1.
        
        # induction
        for t in xrange(len(observations)-2,-1,-1):
            for i in xrange(self.n):
                for j in xrange(self.n):
                    beta[t][i] += self.A[i][j]*self.B_map[j][t+1]*beta[t+1][j]
                    
        return beta
    
    """
    Find the best state sequence (path), given the model and an observation. I.E: max(P(Q|O,model)).
    
    This method is usually used to predict the next state after training. 
    """
    def decode(self, observations):
        # use Viterbi's algorithm. It is possible to add additional algorithms in the future.
        return self._viterbi(observations)
    
    """
    Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
    very similar to the forward-backward algorithm, with the added step of maximization and eventual
    backtracing.
    
    delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
    that generates the highest probability.
    
    psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1), 
    i.e: the previous state.
    
    """
    def _viterbi(self, observations):
        delta = numpy.zeros((len(observations),self.n),dtype=self.precision)
        psi = numpy.zeros((len(observations),self.n),dtype=self.precision)
        
        # init
        for x in xrange(self.n):
            delta[0][x] = self.pi[x]*self.B_map[x][0]
            psi[0][x] = 0
        
        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    if (delta[t][j] < delta[t-1][i]*self.A[i][j]):
                        delta[t][j] = delta[t-1][i]*self.A[i][j]
                        psi[t][j] = i
                delta[t][j] *= self.B_map[j][t]
        
        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = 0 # max value in time T (max)
        path = numpy.zeros((len(observations)),dtype=self.precision)
        for i in xrange(self.n):
            if (p_max < delta[len(observations)-1][i]):
                p_max = delta[len(observations)-1][i]
                path[len(observations)-1] = i
        
        # path backtracing
        path = numpy.zeros((len(observations)),dtype=self.precision)
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        
        return path
    
    """
    Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.
    
    The xi variable is a numpy array indexed by time, state, and state (TxNxN).
    xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
    time 't+1' given the entire observation sequence.
    """    
    def _calcxi(self,observations,alpha=None,beta=None):
        if alpha is None:
            alpha = self._calcalpha(observations)
        if beta is None:
            beta = self._calcbeta(observations)
        xi = numpy.zeros((len(observations),self.n,self.n),dtype=self.precision)
        
        for t in xrange(len(observations)-1):
            denom = 0.0
            for i in xrange(self.n):
                for j in xrange(self.n):
                    thing = 1.0
                    thing *= alpha[t][i]
                    thing *= self.A[i][j]
                    thing *= self.B_map[j][t+1]
                    thing *= beta[t+1][j]
                    denom += thing
            for i in xrange(self.n):
                for j in xrange(self.n):
                    numer = 1.0
                    numer *= alpha[t][i]
                    numer *= self.A[i][j]
                    numer *= self.B_map[j][t+1]
                    numer *= beta[t+1][j]
                    xi[t][i][j] = numer/denom
                    
        return xi

    """
    Calculates 'gamma' from xi.
    
    Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
    in state 'i' at time 't' given the full observation sequence.
    """    
    def _calcgamma(self,xi,seqlen):
        gamma = numpy.zeros((seqlen,self.n),dtype=self.precision)
        
        for t in xrange(seqlen):
            for i in xrange(self.n):
                gamma[t][i] = sum(xi[t][i])
        
        return gamma
    
    """
    Updates this HMMs parameters given a new set of observed sequences.
    
    observations can either be a single (1D) array of observed symbols, or a 2D
    matrix, each row of which is a seperate sequence. The Baum-Welch update
    is repeated 'iterations' times, or until the sum absolute change in
    each matrix is less than the given epsilon.  If given multiple
    sequences, each sequence is used to update the parameters in order, and
    the sum absolute change is calculated once after all the sequences are
    processed.
    """    
    def train(self, observations, iterations=1,epsilon=0.0001,thres=-0.001):
        self._mapB(observations)
        
        for i in xrange(iterations):
            prob_old, prob_new = self.trainiter(observations) # train iter should also update model and cacheB

            if (self.verbose):      
                print "iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new, ", converging =", ( prob_new-prob_old > thres )
                
            if ( abs(prob_new-prob_old) < epsilon ):
                # converged
                break
                
    def _updatemodel(self,new_model):
        self.pi = new_model['pi']
        self.A = new_model['A']
                
    def trainiter(self,observations):
        # call the EM algorithm
        new_model = self._baumwelch(observations)
        
        # calculate the log likelihood of the previous model
        prob_old = self.forwardbackward(observations)
        
        # update the model with the new estimation
        self._updatemodel(new_model)
        
        # map observable probabilities
        self._mapB(observations)
        
        # calculate the log likelihood of the new model
        prob_new = self.forwardbackward(observations)
        
        return prob_old, prob_new
    
    """
    new transitions probability matrix A is set to: expected_transitions(i->j)/expected_transitions(i)
    """
    def _reestimateA(self,observations,xi,gamma):
        A_new = numpy.zeros((self.n,self.n),dtype=self.precision)
        for i in xrange(self.n):
            for j in xrange(self.n):
                numer = 0.0
                denom = 0.0
                for t in xrange(len(observations)-1):
                    numer += (self.eta(t,len(observations)-1)*xi[t][i][j])
                    denom += (self.eta(t,len(observations)-1)*gamma[t][i])
                A_new[i][j] = numer/denom
        return A_new
    
    def _calcstats(self,observations):
        stats = {}
        
        stats['alpha'] = self._calcalpha(observations)
        stats['beta'] = self._calcbeta(observations)
        stats['xi'] = self._calcxi(observations,stats['alpha'],stats['beta'])
        stats['gamma'] = self._calcgamma(stats['xi'],len(observations))
        
        return stats
    
    def _reestimate(self,stats,observations):
        new_model = {}
        
        # new init vector is set to the frequency of being in each step at t=0 
        new_model['pi'] = stats['gamma'][0]
        new_model['A'] = self._reestimateA(observations,stats['xi'],stats['gamma'])
        
        return new_model
        
    
    """
    An EM(expectation-modification) algorithm devised by Baum-Welch. Finds a local maximum
    that outputs the model that produces the highest probability, given a set of observations.
    
    xi[t][i][j] = P(qt=Si, qt+1=Sj|O,model) - the probability of being in state i at time t,
    and in state j at time t+1, given the ENTIRE observation sequence.
    
    gamma[t][i] = sum(xi[i][j]) - the probability of being in state i at time t, given the ENTIRE 
    observation sequence.
    """
    def _baumwelch(self,observations):
        # E step - calculate statistics
        stats = self._calcstats(observations)
        
        # M step
        return self._reestimate(stats,observations)

    def _mapB(self,observations):
        raise NotImplementedError("a mapping function for B(observable probabilities) must be implemented")
        