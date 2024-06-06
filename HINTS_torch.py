# Langevin or Hamiltonian MCMC version: overrides primitive proposal
# Requires pytorch
# Note this is MALA at each leaf proposal ... you could take multiple gradient steps within primitive move and it would be HMC

import torch
import copy

from HINTS import *


class HINTS_HMC(HINTS):
    def __init__(self, args, fn, noise_sd = 1.0): # noise sd is not used
        super().__init__(args, fn)
        self.epsilon = args.epsilon
        self.noise_sd = noise_sd
        self.L = args.L
    #
    # MALA version ... correction depends on knowing gradient at proposed state
    def primitive_move(self, model, index = 0, always_accept = False, verbose = False):
        scenarios = self.scenarios(0, index)
        if verbose:
            print(len(scenarios))
        v = self.fn(model, scenarios, True) # puts a gradient into state as a side effect for HMC
        # as model may point at a cache entry, we should not mutate it
        current = self.fn.sample_initial_state() # empty model - we can mutate this
        correction = 0.0
        momenta = []
        for l in range(self.L):
            if verbose:
                print(l)
            with torch.no_grad(): # don't need gradients of gradients!
                pno = 0 # parameter number
                for f, f_prime in zip(model.parameters(), current.parameters()):
                    if l==0:
                        p_init = torch.randn(f.shape) # sample momentum *for this parameter* 
                        correction -= 0.5 *  (p_init * p_init).sum() # INITIAL Kinetic energy term: subtract from correcion - does not require 1D param
                        if verbose:
                            print("gradient = ", f.grad.data)
                            print("initial correction = ", correction.item())
                        p_init += self.epsilon * 0.5 * f.grad.data # momentum half step
                        momenta.append(p_init)
                        f_prime.data = f.data + self.epsilon * p_init * self.noise_sd # so noise_sd gives an idea of distance in state space (like inverse mass Matrix)
                    else: # l > 0
                        if verbose:
                            print("Gradient shape", f_prime.grad.data.shape)
                        momenta[pno] += self.epsilon * 0.5 * f_prime.grad.data # momentum half step using new gradient from last step                 
                        f_prime.data += self.epsilon * momenta[pno] * self.noise_sd # so noise_sd gives an idea of distance in state space (= inverse mass Matrix)
                        if verbose:
                            print("gradient = ", f_prime.grad.data)
                    if verbose:
                        print("momentum = ", momenta[pno])
                        print("proposal = ", f_prime.data)
                    pno += 1
            # compute the value of the new state and its gradient [in an intermediate HMC step we don't need v_prime but we would run this to calculate the gdt]
            if verbose:
                print("updated:")
                for f_prime in current.parameters():
                    print(f_prime)
                print("counter", self.fn.counter)
            #
            if (l > 0):
                current = copy.deepcopy(current) # creates new object id if it could be in cache; NB deepcopy kills gradients
            current.zero_grad() # ok to mutate this object which is known not to be a cache entry
            #
            v_prime = self.fn(current, scenarios, True) # the current state parameters will acquire a new grdient here (pass in True to calc gradient)
            #
            with torch.no_grad():
                for pno, f_prime in enumerate(current.parameters()):
                    momenta[pno] += self.epsilon * 0.5 * f_prime.grad.data # momentum second half step (stored with prev state)
                    if verbose:
                        print("momentum after second half step = ", momenta[pno])
                    if l == (self.L-1):
                        correction +=  0.5 *  (momenta[pno] * momenta[pno]).sum() # FINAL kinetic energy term of H_new: add to correction
                #
            # standard MHR / HINTS acceptance
        vdiff = (v_prime - v)/self.Ts[0] # PE change ... these are cached evaluations, no side effects
        #correction = 0 # TEMPOARY OVERRDE - SGD
        if verbose:
            print("l = ", l)
            print("v = ", v.item(), "v_prime = ", v_prime.item(), "vdiff = ", vdiff.item(), "correction = ", correction.item())
            print("p(accept) = ", np.exp((vdiff - correction).item()))
        accept = True if always_accept else self.metropolis_accept(vdiff - correction)
        #
        (self.acceptances if accept else self.rejections)[0] += 1
        return((current, vdiff.item()) if accept else (model, 0.0))
        #return((current, vdiff - correction) if accept else (model, 0.0))