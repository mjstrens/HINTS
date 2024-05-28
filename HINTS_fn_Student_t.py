# state and proposal
# NB make sure the state is hashable so the cache works
from HINTS_fn import *
from scipy.stats import t

try:
    import torch
    from torch.distributions import StudentT
    USE_TORCH = True
except Exception as e:
    print("tried loading torch but got exception", e)
    print("using scipy (slower than torch)")
    USE_TORCH = False

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


# sigma must now be bound by TestFnT
def proposalT(state, index=0, sigma = 1000.0): # index is not used here but the arg must be listed for HINTS
    new_nu = state.nu + randint(-1,+2) # displace by -1, 0 or 1
    new_mu = state.mu + sigma * randn() # MJAS divided by 4
    new_tau = state.tau * np.exp(sigma * randn()) # random walk on log value    
    if new_nu == 0: # reflecting barrier at 1 d.o.f.
        new_nu = 2
    return(StateT(new_nu, new_mu, new_tau), 0.0) # construct a new state; symmetrical so Hastings correction (delta) is zero


# create the user function for HINTS to work with this

# version that takes 2D data of size NUM_SCENARIOS x LEAF_SIZE
class TestFnT(UserFn):
    def __init__(self, data, proposal_sigma, additive = True):
        self.N = data.shape[0]
        self.per_leaf = data.shape[1]
        self.data = data
        self.additive = additive
        super().__init__(lambda state, index: proposalT(state, index, proposal_sigma), additive) # bind the sigma
        #
    def sample_initial_state(self, nu, mu, tau, runs = 1):
        nus = nu + randint(0, 30, runs)
        # make sure any nu value is at least 1
        nus = np.where(nus < 1, 1, nus)
        mus = mu + 0.25 * randn(runs)
        taus = tau + 0.25 * randn(runs)
        return([StateT(nus[i], mus[i], taus[i]) for i in range(runs)])
        #
    def evaluate(self, state, term_index, with_gradient = False):
        ll = self.cached_eval_fast(state.nu, state.mu, state.tau, term_index)
        return (ll if self.additive else ll * self.N) # NB gets divided thru by the number in the evaluation
        #
    @lru_cache(maxsize = 1000000) 
    def cached_eval_fast(self, nu, mu, tau, term_index):# simple args so caching works through assigned/returned states
        self.counter += self.per_leaf
        if USE_TORCH:
            return(StudentT(nu, mu, tau).log_prob(torch.Tensor(self.data[term_index])).sum())
        else:
            return(t(nu, loc = mu, scale = tau).logpdf(self.data[term_index]).sum())