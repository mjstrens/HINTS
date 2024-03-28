# version that does not assume higher level evaluations are compositions (sum of logs) of lower level evaluations
# no caching or reuse

# fewer dependencies
import numpy as np

class UserFnSimple:
    def __init__(self, proposal):
        self.counter = 0 # keeps track of term evaluations
        self.total_counter = 0 # this includes cached ones
        self.proposal = proposal # user provides a proposal function
    #
    # where should the chain start
    def sample_initial_state(self):
        pass # user must implement       
    #
    # user to override this - e.g. with PF likelihood
    def evaluate_sequence(self, state, term_index, num_terms):
        pass # user must implement (term in additive structure)
    #
    #
    # scenarios are usually a contiguous sequence when there is state dependency so an evaluation of 1:50 is not the same as a 1:25 and 25:50 combined
    def __call__(self, state, scenarios):
        n = len(scenarios)
        sum_f = self.evaluate_sequence(state, min(scenarios), n) # must be a contiguous range but need not be ordered
        self.total_counter += n # lru cache ignores this side effect
        return(sum_f)

# simple test where there is a sequential dependency like in a PF (can't compose evaluations)
# IOU process with decay parameter alpha as the state to be estimated
class TestFnIOU(UserFnSimple):
    def __init__(self, proposal, N, alpha, stochastic = True):
        self.N = N
        self.stochastic = True
        self.alpha = alpha # the target for inference (IOU decay param)
        super().__init__(proposal)
        self.ys = self.make_observations(N, alpha) # simulate some data [could use a fixed seed for this too]
        self.lockdown_randomness(0)
        #
    # evaluations must be repeatable between calls to lockdown_randomness
    # (typically call this each time we return to root)
    def lockdown_randomness(self, seed):
        rng = np.random.default_rng(seed) # a Generator with this seed
        self.initial_estimate = rng.standard_normal(self.N) # we won't use all of these if leaf size is >1, so could be a bit more efficient
    #
    def make_observations(self, N, alpha): # unit Normal samples length N
        ys = np.zeros(N)
        noise = np.random.randn(N)
        y = 0.0
        for i in range(N):
            ys[i] = ((1.0 - alpha) * y) + noise[i]
            y = ys[i]
        return(ys)
    #
    def sample_initial_state(self): # initial guess for alpha in HINTS (nothing to do with the PF inference) - we do it here because a suitable prior is problem-dependent
        return(np.zeros(1))
    #
    # inference of decay param on an IOU process (proxy for a particle filter)
    def evaluate_sequence(self, alpha, term_index, num_terms):
        # alpha is the param we want to infer
        log_like = 0.0 # accumulate log likelihood (negative qty)
        if self.stochastic:
            y_prev = self.initial_estimate[term_index]
        else:
            y_prev = 0.0
        for i in range(term_index, term_index + num_terms):
            expected_y = (1.0 - alpha) * y_prev
            y = self.ys[i]
            z = y - expected_y
            # log of normal pdf for this transition of the IOU process
            log_like -= 0.5 * z * z # we can ignore the log sigma root 2 pi term which is constant here
            y_prev = y
        return(log_like)
    