import numpy as np
import pandas as pd
import copy
from HINTS import *
from HINTS_fn_Student_t import *

def run_t_sampler(args):
    seed(4) # just happens to give us data where MLL nu is actual nu
    data = t.rvs(args.nu, loc = args.mu, scale = args.tau, size =[args.NUM_SCENARIOS, args.LEAF_SIZE])
    g = TestFnT(data, proposal_sigma = args.proposal_sigma, additive = args.additive)
    if not args.additive:
        print("Using AVERAGER")
    hmc = HINTS(args, g)
    np.random.seed(args.id)
    state  = g.sample_initial_state(1, 1, 1)[0]
    print(args.id, "initial state", state)
    # include initial state in history
    history = []
    hstate = copy.deepcopy(state)
    history.append({'nu':hstate.nu, 'mu':hstate.mu, 'tau':hstate.tau, 'acceptances':copy.deepcopy(hmc.acceptances), 'rejections':copy.deepcopy(hmc.rejections), 'evals_cache':hmc.fn.counter, 'evaluations':hmc.fn.total_counter})    
    for tstep in range(args.iterations):
        hmc.shuffle()
        state, correction = hmc.hints(state, args.levels) # e.g. dbg = (t==0)
        # show progress
        if ((tstep%100)==99):
            print(tstep+1, hmc.acceptances, hmc.rejections, hmc.fn.total_counter, hmc.fn.counter)
        #
        hstate = copy.deepcopy(state)
        history.append({'nu':hstate.nu, 'mu':hstate.mu, 'tau':hstate.tau, 'acceptances':copy.deepcopy(hmc.acceptances), 'rejections':copy.deepcopy(hmc.rejections), 'evals_cache':hmc.fn.counter, 'evaluations':hmc.fn.total_counter})    
    return(pd.DataFrame(history))