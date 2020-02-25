from numba import jit
import math
import numpy as np
import time
from numpy.random import seed, randn, rand, shuffle
#from line_profiler import LineProfiler

from functools import lru_cache # prefer diskcache because it can memoize unhashable types

# diskcache is great at memoizing but slow for fine grain calls
#from diskcache import Cache
#cache = Cache(size_limit=int(1e9)) #1GB
#cache.stats(enable=True)

# suppose either the state is a hashable type
# or it is a user class instance and the proposal mechanism always makes a deepcopy
class HashableItem():
    def __init__(self, x):
        self.x = x


# template for user function
# old version uses a trick to make sure states are hashable ... so make sure you do not modify them in place!!
# new version relies on diskcache which does this by magic
class UserFn:
    def __init__(self, proposal):
        self.counter = 0 # keeps track of term evaluations
        self.total_counter = 0 # this includes cached ones
        self.proposal = proposal # user provides a proposal function
        #
        # (care must be taken: any side effects of the calculation [autograd] are not preserved by caching)
    # must have no side effects: count evaluations using cache stats instead of counter
    # as first arg is self, that must have no modifiable state
    #@cache.memoize()
    def cached_evaluate(self, state, term_index):
        return(self.user_evaluate(state, term_index))
    # User may either use this caching support or implement evaluate() directly
    def evaluate(self, state, term_index):
        self.total_counter +=1
        return(self.cached_evaluate(state, term_index))
    def sample_initial_state(self):
        pass # user must implement       
    # return sum over scenarios of log density term
    # this can be computed in parallel across scenarios, but that would restrict cache benefit
    # so better to use minibatch at the single scenario level (self.evaluate)
    # state could be a numpy array, or a complex structure such as a pyTorch model (state dict)
    # level is supplied in case we want to compute a gradient at level 0 
    def __call__(self, state, scenarios):
        n = len(scenarios)
        sum_f  = sum([self.evaluate(state, term) for term in scenarios])
        sum_f += self.evaluate_regularisation(state) * n # OPTIONAL, MUST scale with number of scenarios
        self.total_counter += n # lru cache ignores this side effect
        return(sum_f if self.additive else sum_f/n)
        #            
    def evaluate_regularisation(self, state):
        return(0.0) # none by default, user can override        

# Simple test function: Gaussian with per term bias
class TestFn(UserFn):
    def __init__(self, proposal, N, additive = False):
        self.N = N
        self.additive = additive
        super().__init__(proposal)
        #
    def sample_initial_state(self):
        return(randn(1))
        #
    # remember to clear the cache (by deleting the file) when you change this function!!
    # optionally you can specify an evaluate() function instead, in which case diskcache will be bypassed
    # DANGER: THIS FUNCTION MUST HAVE NO SIDE EFFECTS TO BE CACHED!!!
    #def user_evaluate(self, state, term_index):
    # in this case the simple version with no caching is much faster
    # NB it may be faster for the user to implement caching at the primitive level to avoid momoizing self
    def evaluate(self, state, term_index):
        # at this point the user should ensure args are Hashable if planning to use lru_cache for repeat calls
        # we use lru_cache in preference to disk cache when the hashing overhead is important
        # but prefer diskcache when persistence is needed
        # can use HashableItem here if passing (say) a big numpy array
        #return(cached_eval_fast(HashableItem(state), term_index, self.N, self.additive))
        return(self.cached_eval_fast(state[0], term_index, self.N, self.additive)) # first arg is now float, so hashable
    # option to cache at most primitive level
    @lru_cache(maxsize = 1000000)    
    def cached_eval_fast(self, x, term_index, N, additive):
        self.counter += 1 # lru cache ignores this side effect
        return(eval_fast(x, term_index, N, additive))
        
@jit(nopython=True)
def eval_fast(x, term_index, N, additive):
    noise = float((term_index * 2 + 1) - N)/float(N) # generally we apply variance reduction to noise if possible (essential reparameterisation if cached)
    v = - 0.5 * x * x + noise * np.sin(x) # non homogeneous noise
    return((v/float(N)) if additive else v) # scale factor is to make results comparable for additive (= classify) vs expectation  
    

