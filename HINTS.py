# the HINTS class
import math
import numpy as np
from numpy.random import seed, randn, rand, shuffle
import random

class HINTS:
    # TO DO move langevin out?
    # Mar 2024: added random shuffle as we go down tree (but not within leaves)
    def __init__(self, args, fn, shuffle_as_we_go = False):
        self.args = args
        self.fn = fn # user fn
        self.shuffle_as_we_go = shuffle_as_we_go
        if 'skip_after_accept' in args:
            self.skip_after_accept = args.skip_after_accept
        else:
            self.skip_after_accept = False
        self.levels = args.levels
        self.design = args.design
        self.levels = args.design.shape[0] - 1
        if 'Ts' in args:
            self.Ts = args.Ts
        else:
            self.Ts = np.flip(np.array([self.args.T + l * self.args.dT for l in range(args.levels + 1)]))
        self.ns = np.cumprod(args.design) # number of scenarios at each level
        self.N = self.ns[args.levels] 
        self.downsample = 1
        if 'downsample' in args:
            self.downsample = args.downsample
        self.reset()
        # Diagnostics:
        print(self.levels)
        print(self.ns)
        print(self.N)
        print(self.Ts)
        # TO DO: move into HINTS class

    # calculate the terms at this level and index
    def scenarios(self, level, index):
        return([self.fixed_scenarios[self.ns[level] * index + i] for i in range(self.ns[level])])
    
    def reset(self):
        print("RESET")
        self.rejections = np.zeros(self.levels + 1, dtype = int)
        self.acceptances = np.zeros(self.levels + 1, dtype = int)
        self.fixed_scenarios = np.arange(self.ns[self.levels])
        
    def shuffle(self):
        shuffle(self.fixed_scenarios)
    
    def metropolis_accept(self, exp_diff):
        return(True if (exp_diff > 0.0) else (False if (exp_diff < -100.0) else (rand() < math.exp(exp_diff))))
    
    def hints(self, state, level, index = 0, always_accept = False): # main function, requires value of current as input
        if (level == 0): return(self.primitive_move(state, index, always_accept)) # this is only separate so we can override
        scenarios = self.scenarios(level, index)
        correction = 0.0
        current = state # we hold state at all levels in the hierarchy
        branches = list(range(self.design[level]))
        if self.shuffle_as_we_go:
            random.shuffle(branches)
        for bi, b in enumerate(branches):
            if (bi % self.downsample) == 0:
                current, delta_correction = self.hints(current, level-1, index * self.design[level] + b) # recursive call
                correction += delta_correction
                if self.skip_after_accept:
                    if correction != 0.0:
                        print("skip back up", bi)
                        break
                    
        # now do composite evaluations AFTER primitive ones, in case primitive ones needed gradients
        vdiff = (self.fn(current, scenarios) - self.fn(state, scenarios))/self.Ts[level] # these are cached evaluations, no side effects
        accept = True if always_accept else self.metropolis_accept(vdiff - correction) # NB second expression will not be evaluated if always accept
        (self.acceptances if accept else self.rejections)[level] += 1
        return((current, vdiff) if accept else (state, 0.0))
    
    # separate out level zero in case we want to override it (e.g. with HMC)
    def primitive_move(self, state, index = 0, always_accept = False):
        scenarios = self.scenarios(0, index)
        v = self.fn(state, scenarios) # could put a gradient into state as a side effect for HMC
        current, correction = self.fn.proposal(state, index) # need to pass level 0 scenario [=index] in case proposal depends on scenario
        v_prime = self.fn(current, scenarios)
        vdiff = (v_prime - v)/self.Ts[0] # these are cached evaluations, no side effects
        accept = True if always_accept else self.metropolis_accept(vdiff - correction)
        (self.acceptances if accept else self.rejections)[0] += 1
        return((current, vdiff) if accept else (state, 0.0))
        
# TO DO what are the exact criteria on the lower level functions for the proof to hold?