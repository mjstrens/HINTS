import math
import numpy as np
from numpy.random import seed, randn, rand, shuffle
from functools import lru_cache #optional

# suppose either the state is a hashable type
# or it is a user class instance and the proposal mechanism always makes a deepcopy
class HashableItem():
    def __init__(self, x):
        self.x = x


# template for user function
# uses a trick to make sure states are hashable ... so make sure you do not modify them in place!!
class UserFn:
        def __init__(self, proposal, state_is_hashable = False):
            self.counter = 0 # keeps track of term evaluations
            self.total_counter = 0 # this includes cached ones
            self.user_proposal = proposal
            self.wrap = (lambda x:x) if state_is_hashable else (lambda x: HashableItem(x))
            self.unwrap = (lambda x:x) if state_is_hashable else (lambda hx: hx.x)

        def sample_initial_state(self):
            return(self.wrap(self.user_sample_initial_state()))
        
        # wrap user proposal function for HashableItem
        def proposal(self, hstate, term_index):
            x1,c1 = self.user_proposal(self.unwrap(hstate), term_index)
            return(self.wrap(x1),c1) # this object has a new unique ID 
        
        def user_sample_initial_state(self):
            pass # user must implement       
               
        # return sum over scenarios of log density term
        # this can be computed in parallel across scenarios, but that would restrict cache benefit
        # so better to use minibatch at the single scenario level (self.evaluate)
        # state could be a numpy array, or a complex structure such as a pyTorch model (state dict)
        # level is supplied in case we want to compute a gradient at level 0 
        def __call__(self, state, scenarios, level):
            n = len(scenarios)
            sum_f  = sum([self.evaluate(state, term, level) for term in scenarios])
            sum_f += self.evaluate_regularisation(state) * n # OPTIONAL, MUST scale with number of scenarios
            return(sum_f if self.additive else sum_f/n)
                    
        def evaluate_regularisation(self, state):
            return(0.0) # none by default, user can override        

        # User may either use this caching support or implement evaluate() directly
        def evaluate(self, state, term_index, level):
            self.total_counter += 1 
            return(self.cached_evaluate(state, term_index, level))
        
        # (care must be taken: any side effects of the calculation [autograd] are not preserved by caching)
        @lru_cache(maxsize=1048576) # avoids duplicate evaluations 
        def cached_evaluate(self, state, term_index, level):
            return(self.user_evaluate(self.unwrap(state), term_index, level))

        # Only needed to support cached implementation; otherwise user can implememt evaluate directly
        def actual_evaluate(self, state, term_index, level):
            self.counter += 1

# Simple test function: Gaussian with per term bias
class TestFn(UserFn):
        def __init__(self, proposal, N, additive = False):
            self.N = N
            self.additive = additive
            super().__init__(proposal)
        
        def user_sample_initial_state(self):
            return(randn(1))
                
        def user_evaluate(self, state, term_index, level):
            # NB if this uses a controlled random number, ensure not to interfere with HINTS 
            # simple 1d function
            noise = float((term_index * 2 + 1) - self.N)/float(self.N) # generally we apply variance reduction to noise if possible
            v = - 0.5 * state[0] * state[0] + noise * math.sin(state[0]) # non homogeneous noise
            self.counter += 1 # you can do this yourself rather than call the parent evaluate func
            return((v/float(self.N)) if self.additive else v) # scale factor is to make results comparable for additive (= classify) vs expectation  
        

