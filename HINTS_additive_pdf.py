from numba import jit
import math
import numpy as np
import time
from numpy.random import seed, randn, rand, shuffle
from functools import lru_cache # prefer diskcache because it can memoize unhashable types




#template for user function that is an averager in probability, not log probability
class UserFnAdditive:
    def __init__(self, proposal):
        self.proposal = proposal # user provides a proposal function
        self.counter = 0 # keeps track of term evaluations
        self.total_counter = 0 # this includes cached ones
    #
    # where should the chain start
    def sample_initial_state(self):
        pass # user must implement       
    #
    # the low level evaluation
    # with_gradient is for systems like pytorch that can hang a gradient onto the quantities returned
    def evaluate_prob(self, state, term_index):
        pass # user must implement (term in additive structure)
        # user should increment self.counter for cache misses
    #
    # return log of subset target
    def __call__(self, state, scenarios):
        n = len(scenarios)
        sum_f  = sum([self.evaluate_prob(state, term) for term in scenarios]) 
        #reg =  self.evaluate_regularisation(state, with_gradient) # OPTIONAL
        self.total_counter += n # lru cache ignores this side effect
        return(np.log(sum_f/n)) # note the np.log ... so HINTS still see log probs
        #            
    #def evaluate_regularisation(self, state, with_gradient = False):
    #    return(0.0) # none by default, user can override        

    
# for caching (now the user's responsibility), either the state is a hashable type
# or it is a user class instance and the proposal mechanism always makes a dee pcopy

# stratified sample to avoid extreme noise in stepped Gaussian
def stratify(wts):
    n = wts.size
    return(np.array([np.argmin(readout > np.cumsum(wts)) for readout in np.arange(1.0/(n+n), 1.0, 1.0/n)]))

#stratify(np.array([0.01, 0.99, 0.0]))    

eps = 1e-10    
# Simple test function: Gaussian with per term bias
class TestFnStepped(UserFnAdditive):
    def __init__(self, proposal, N,  reduce_variance = True):
        self.N = N
        sub_data = randn(N,2)
        self.l = np.min(sub_data, axis = 1)
        self.u = np.max(sub_data, axis = 1)
        if reduce_variance: # blur the distribution at spikes that cannot be represented accurately
            self.l -= 10.0/N
            self.u += 10.0/N            
        super().__init__(proposal)
        #
    def sample_initial_state(self):
        return(randn(1))
        #
    def evaluate_prob(self, state, term_index):
        return(self.cached_eval_prob_fast(state[0], term_index))
    # option to cache at most primitive level
    @lru_cache(maxsize = 1000000)    
    def cached_eval_prob_fast(self, x, i):
        self.counter += 1 # lru cache ignores this side effect
        return(eval_prob_fast(x, self.l[i], self.u[i]))

@jit(nopython=True)
def eval_prob_fast(x, l, u):
    return(eps + (1.0 - eps) * np.logical_and(x >= l, x <= u)/(u - l))  # Indicator function lower bounded by eps

    


