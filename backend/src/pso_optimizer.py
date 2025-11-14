from pyswarm import pso

def optimize_threshold(cost_function, lb, ub):
    best_threshold, _ = pso(cost_function, lb, ub, swarmsize=10, maxiter=10)
    return best_threshold
