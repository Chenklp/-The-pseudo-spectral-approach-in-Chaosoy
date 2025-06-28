import numpy as np
import chaospy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

from typing import Union, Optional
import numpy.typing as npt
# if you want you can rely also on already implemented Oscillator class
# from utils.oscillator import Oscillator


# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(init_cond: tuple[float, float], t: float, args: tuple[float, float, float, float]) -> list[float]:
    """Defines the system of ODEs for the damped oscillator."""
    x1, x2 = init_cond
    c, k, f, w = args
    f = [x2, f * np.cos(w * t) - k * x1 - c * x2]
    return f

def discretize_oscillator_odeint(model, atol: float, rtol: float, init_cond: tuple[float, float], 
                                 args: tuple[float, float, float, float], 
                                 t: np.ndarray, t_interest: int) -> float:
    """Solves the ODE system and returns the solution at a specified time index."""
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    return sol[t_interest, 0]

if __name__ == '__main__':
    ### deterministic setup ###

    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.0
    # initial conditions setup
    init_cond   = y0, y1
    # model_kwargs = {"c": c, "k": k, "f": f}  # if you want to use the Oscillator class, you can uncomment this line
    # init_cond = {"y0": y0, "y1": y1}  # if you want to use the Oscillator class, you can uncomment this line

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t_grid          = np.array([i*dt for i in range(grid_size)])
    #t_grid = np.arange(0, t_max + dt, dt)
    t_interest  = -1

    ### stochastic setup ####
    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05
    # TODO: create uniform distribution object
    distr_w = cp.Uniform(w_left, w_right)
    
    # the truncation order of the polynomial chaos expansion approximation
    Ns = [1, 2, 3, 4, 5, 6]
    # the quadrature degree of the scheme used to computed the expansion coefficients
    Ks = [1, 2, 3, 4, 5, 6]
    
    assert(len(Ns)==len(Ks))

    # vector to save the statistics

    
    # Reference Monte Carlo solution
    E_ref = -0.43893703
    V_ref = 0.00019678
    
    # Arrays to store results
    exp_m = np.zeros(len(Ns))
    var_m = np.zeros(len(Ns))
    exp_cp = np.zeros(len(Ns))
    var_cp = np.zeros(len(Ns))
    rel_err_mean_m = np.zeros(len(Ns))
    rel_err_var_m = np.zeros(len(Ns))
    rel_err_mean_cp = np.zeros(len(Ns))
    rel_err_var_cp = np.zeros(len(Ns))
    
    # perform polynomial chaos approximation + the pseudo-spectral
    for idx, (N, K) in enumerate(zip(Ns, Ks)):
        # Generate orthogonal polynomials and quadrature nodes
        poly = cp.generate_expansion(N, distr_w)
        nodes, weights = cp.generate_quadrature(K, distr_w, rule='G')
        weights = weights.flatten()
        nodes_flat = nodes.flatten()
        
        print(f"K={K}, N={N}: Using {len(nodes_flat)} quadrature nodes")
        
        # Evaluate the model at all quadrature nodes
        model_evals = np.zeros(len(nodes_flat))
        start_time = time.time()
        for i, omega in enumerate(nodes_flat):
            args_i = (c, k, f, omega)
            model_evals[i] = discretize_oscillator_odeint(
                model, atol, rtol, init_cond, args_i, t_grid, t_interest
            )
        eval_time = time.time() - start_time
        print(f"Model evaluations completed in {eval_time:.2f} seconds")
        
        # TODO: perform polynomial chaos approximation + the pseudo-spectral approach manually
        basis_evals = poly(*nodes)
        norms_sq = np.sum(weights * basis_evals**2, axis=1)
        alpha_m = np.zeros(len(basis_evals))
        for k in range(len(alpha_m)):
            numerator = np.sum(weights * basis_evals[k] * model_evals)
            alpha_m[k] = numerator / norms_sq[k]
        
        # Manual moments
        exp_m[idx] = alpha_m[0]
        var_m[idx] = np.sum(alpha_m[1:]**2 * norms_sq[1:])
        
        # TODO: perform polynomial chaos approximation + the pseudo-spectral approach using chaospy
        alpha_cp = cp.fit_quadrature(poly, nodes, weights, model_evals)
        exp_cp[idx] = cp.E(alpha_cp, distr_w)  # Fixed: pass distr_w instead of poly
        var_cp[idx] = cp.Var(alpha_cp, distr_w)  # Fixed: pass distr_w instead of poly
        
        print(f"Completed K={K}, N={N}: Manual mean={exp_m[idx]:.8f}, Chaospy mean={exp_cp[idx]:.8f}")
    
    # Compute relative errors with respect to the reference solution
    rel_err_mean_m = np.abs(exp_m - E_ref) / np.abs(E_ref)
    rel_err_var_m = np.abs(var_m - V_ref) / V_ref
    rel_err_mean_cp = np.abs(exp_cp - E_ref) / np.abs(E_ref)
    rel_err_var_cp = np.abs(var_cp - V_ref) / V_ref
    
    print('\nMean Comparison:')
    print('K | N | Manual             | ChaosPy            | Manual Rel Err     | ChaosPy Rel Err')
    for idx in range(len(Ns)):
        print(f'{Ks[idx]} | {Ns[idx]} | {exp_m[idx]:<18.8f} | {exp_cp[idx]:<18.8f} | '
              f'{rel_err_mean_m[idx]:<18.8e} | {rel_err_mean_cp[idx]:<18.8e}')
    
    print('\nVariance Comparison:')
    print('K | N | Manual             | ChaosPy            | Manual Rel Err     | ChaosPy Rel Err')
    for idx in range(len(Ns)):
        print(f'{Ks[idx]} | {Ns[idx]} | {var_m[idx]:<18.8f} | {var_cp[idx]:<18.8f} | '
              f'{rel_err_var_m[idx]:<18.8e} | {rel_err_var_cp[idx]:<18.8e}')

    # Plot the convergence of errors
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(Ks, rel_err_mean_m, 'o-', label='Manual')
    plt.plot(Ks, rel_err_mean_cp, 's-', label='ChaosPy')
    plt.yscale('log')
    plt.xlabel('Quadrature Degree (K)')
    plt.ylabel('Relative Error')
    plt.title('Relative Error in Mean Estimation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(Ks, rel_err_var_m, 'o-', label='Manual')
    plt.plot(Ks, rel_err_var_cp, 's-', label='ChaosPy')
    plt.yscale('log')
    plt.xlabel('Quadrature Degree (K)')
    plt.ylabel('Relative Error')
    plt.title('Relative Error in Variance Estimation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('assignment_4.png')
    plt.show()