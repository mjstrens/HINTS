# Langevin or Hamiltonian MCMC version: overrides primitive proposal
# Requires pytorch

import torch

from HINTS import *


class HINTS_HMC(HINTS):
    def __init__(self, args, fn, noise_sd = 0.01):
        super().__init__(args, fn)
        self.epsilon = args.epsilon
        self.noise_sd = noise_sd
    #
    # HMC version ... correction depends on knowing gradient at proposed state
    def primitive_move(self, model, index = 0, always_accept = False):
        scenarios = self.scenarios(0, index)
        #print("primitive", index, len(scenarios))
        v = self.fn(model, scenarios, True) # puts a gradient into state as a side effect for HMC
        current = self.fn.sample_initial_state() # empty model
        # do this with no grad...
        correction = 0.0
        for f, f_prime in zip(model.parameters(), current.parameters()):
            # f.grad.data is the right shape to store momentum temporarily
            ###TO DO how big is f.grad.data compared with proposal noise
            f.grad.data = self.noise_sd * torch.randn(f.shape).to(f.device) + self.epsilon * 0.5 * f.grad.data # TO DO check gradient convention
            f_prime.data = f.data + 0.5 * self.epsilon * f.grad.data # add momentum
            correction -= 0.5 * (f.grad.data * f.grad.data).sum() # Kinetic energy term of -H
        # compute the value of the new state and its gradient
        v_prime = self.fn(current, scenarios, True) # need gdt again
        # store the new momentum in the grad entries of the candidate model
        for f, f_prime in zip(model.parameters(), current.parameters()):
            p_prime = f.grad.data + self.epsilon * 0.5 * f_prime.grad.data # TO DO check gradient convention
            correction += 0.5 * (p_prime * p_prime).sum() # kinetic energy term of H_new
            f.grad = None # must not reuse (unless we do more leapfrog steps)
            f_prime.grad = None
        
        # standard MHR / HINTS acceptance
        vdiff = (v_prime - v)/self.Ts[0] # PE change ... these are cached evaluations, no side effects
        #correction = 0 # TEMPOARY OVERRDE - SGD
        accept = True if always_accept else self.metropolis_accept(vdiff - correction)
        (self.acceptances if accept else self.rejections)[0] += 1
        return((current, vdiff) if accept else (model, 0.0))