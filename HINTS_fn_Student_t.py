# state and proposal
# NB make sure the state is hashable so the cache works
from HINTS_fn import *
from scipy.stats import t

class StateT:
    def __init__(self, nu, mu = 0.0, tau = 1.0):
        self.nu = nu
        self.mu = mu
        self.tau = tau
        #self.distrib = t(self.nu, loc = self.mu, scale = self.tau) # faster version - but less caching benefit?
    def log_like(self, xs):
        #return(self.distrib.logpdf(xs).sum())
        return(t(self.nu, loc = self.mu, scale = self.tau).logpdf(xs).sum())
    def __str__(self):
        return("Student t state variable: nu = {},  mu = {:.3f}, tau = {:.3f}".format(self.nu, self.mu, self.tau))


def proposalT(state, index=0): # index is not used here but the arg must be listed for HINTS
    new_nu = state.nu + randint(-1,+2) # displace by -1, 0 or 1
    new_mu = state.mu + 0.05 * randn()
    new_tau = state.tau * np.exp(0.05 * randn()) # random walk on log value    
    if new_nu == 0: # reflecting barrier at 1 d.o.f.
        new_nu = 2
    return(StateT(new_nu, new_mu, new_tau), 0.0) # construct a new state; symmetrical so Hastings correction (delta) is zero


# create the user function for HINTS to work with this

# version that takes 2D data of size NUM_SCENARIOS x LEAF_SIZE
class TestFnT(UserFn):
    def __init__(self, data):
        self.N = data.shape[0]
        self.per_lead = data.shape[1]
        self.data = data
        super().__init__(proposalT)
        #
    def sample_initial_state(self):
        return(StateT(1, 0.0, 1.0)) # start with fattest distro
        #
    def evaluate(self, state, term_index, with_gradient = False):
        return(self.cached_eval_fast(state.nu, state.mu, state.tau, term_index))
        #
    @lru_cache(maxsize = 1000000) 
    def cached_eval_fast(self, nu, mu, tau, term_index):# simple args so caching works through assigned/returned states
        self.counter += 1
        return(t(nu, loc = mu, scale = tau).logpdf(self.data[term_index]).sum())