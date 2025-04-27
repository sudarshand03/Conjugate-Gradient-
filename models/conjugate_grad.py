import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time

def CG(A, b, tol, nmax, verb=True):
    '''
    Conjugate Gradient Method
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    verb - verbose output flag
    Outputs:
    x - solution
    iterates - the iterates calculated to find the solution
    i - the number of iterations taken
    '''
    # Get true solution for error computation
    x_star = np.linalg.solve(A, b)
    
    # Initialize
    x = np.zeros_like(b)  # Start from zero initial guess
    iterates = np.zeros([nmax, b.size])
    iterates[0,:] = x.flatten()
    r = b - A @ x  # Initial residual
    p = r.copy()   # Initial search direction
    
    # Iteration
    for i in range(nmax):
        # Store current iterate
        iterates[i, :] = x.flatten()
        
        # Compute step length
        Ap = A @ p
        alpha = float((r.T @ r) / (p.T @ Ap))
        
        # Update solution and residual
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
        # Check convergence
        if norm(r_new) < tol:
            if verb:
                print(f'Converged in {i+1} iterations')
            return x_new, iterates[:i+1, :], i+1, x_star
        
        # Compute beta for conjugate direction
        beta = float((r_new.T @ r_new) / (r.T @ r))
        
        # Update search direction
        p = r_new + beta * p
        
        # Prepare for next iteration
        x = x_new
        r = r_new

    # If maximum iterations reached
    if verb:
        print('Maximum Number of iterations exceeded')
    return x, iterates[:i+1, :], i+1, x_star

def driver():
    
    
    
    # Convergence Experiments
    print('Testing convergence of the Conjugate Gradient method')
    K = 100
    rate_n = []
    

    # Experiment for changing n and keeping K constant
    plt.figure(figsize=(10, 6))
    for n in [10,100,1000]:
        # create a symmetric positive definite matrix
        Q, R = np.linalg.qr(np.random.randn(n,n))
        D = np.diag(np.linspace(1,K,n))
    
        A = Q @ D @ Q.T
        b = np.random.randn(n,1)
        b = b / norm(b)

        x, iterates, it_count, x_star = CG(A, b, 1e-8, 5000)

        # Compute errors relative to true solution
        diffs = iterates - x_star.reshape(1,-1)
        err = np.array([norm(diff)/norm(x_star) for diff in diffs])
        rate_n.append(np.average(err[1:] / err[:-1]))
        plt.semilogy(err, label=f'n = {n}')
    
    plt.ylabel('Relative Error')
    plt.xlabel('Iteration')
    #plt.title('CG Convergence for Different Problem Sizes (K=100)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'Average convergence rates for n = [10, 100, 1000]: {rate_n}')

    # Experiment for changing K and keeping n constant
    for n in [10,100,1000]:
        plt.figure(figsize=(10, 6))
        rates_K = []
        for K in [10, 100, 1000, 10000]:
            # create a symmetric positive definite matrix
            Q, R = np.linalg.qr(np.random.randn(n,n))
            D = np.diag(np.linspace(1,K,n))
        
            A = Q @ D @ Q.T
            b = np.random.randn(n,1)
            b = b / norm(b)

            x, iterates, it_count, x_star = CG(A, b, 1e-8, 5000, verb=False)

            # Compute errors relative to true solution
            diffs = iterates - x_star.reshape(1,-1)
            err = np.array([norm(diff)/norm(x_star) for diff in diffs])
            rates_K.append(np.average(err[1:] / err[:-1]))
            plt.semilogy(err, label=f'K = {K}')
            
        plt.ylabel('Relative Error')
        plt.xlabel('Iteration')
        #plt.title(f'CG Convergence for Different Condition Numbers (n={n})')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f'Average convergence rates for n={n}, K=[10,100,1000,10000]: {rates_K}')

if __name__ == "__main__":
    driver()
