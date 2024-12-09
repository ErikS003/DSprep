#Transition-matrices:
# P = [[p(1,1), p(1,2), ... , p(1,n)],
#      [p(2,1), p (2,2), ... , p(2,n)]
#      [             ...             ]
#      [p(n,1), p(n,2), ... , p(n,n)]]
# where p(i,j) is the probability of moving to j given that we are stationed at i
# rows of P will should sum to 1.
# not true in general for columns.
# P is square (NxN)
# rows represent NOW, columns represent NEXT
# P must list all possible states in the state space S
# 
# 
# The probability distribution of X_k = (pi.T)P^k
# where pi is the distribution of X_0, i.e pi = [p(X_0=1),p(x_0=2), ... , p(X_0=n)] 


# if it is possiblt o get form any state to any other state, the markov chian is irreducible


# a state s is periodic if you can only return to s in multiples of an integer n
# example : consider the markov chain given by states A,B,C
# the states are connected by a cycle (A->B->C->A)
# if we start at A, we can only return to A in (3,6,9 ...) steps
# i.e k steps where kmod3=0, where k would also be the "period"
# if a state can be returned to in irregular intervals, it is aperiodic.
# for a markov chain to converge to a unique stationary distribution, the chain must be both irreducible and aperiodic

# the stationary distribution of a markov chain is given by pi*P=pi where pi = [pi_1,pi_2, ... , pi_n]
# finding the stationary distribution is equivalent to solving the system of equations:
# pi_1*p_(1,1)+pi_2*p(2,1)... + pi_n*p_(n,1)=pi_1
# pi_1*p_(1,2)+pi_2*p(2,2)... + pi_n*p_(n,2)=pi_2
#                      ... 
# with sum(pi) = 1
# this is equivalent to the matrix problem (P^T-I)pi=0


import numpy as np
def stationary_distribution(P):
    """
    Finds the stationary distribution of a Markov chain.
    
    Parameters:
        P (numpy.ndarray): The transition matrix (n x n).
        
    Returns:
        numpy.ndarray: The stationary distribution (1 x n).
    """
    n = P.shape[0]
    # Transpose the transition matrix
    P = P.T
    
    # Set up the linear system: (P - I)pi= 0
    A = P - np.eye(n)
    
    # Add the normalization condition: sum(pi) = 1
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n)
    b = np.append(b, 1)
    
    # Solve for pi
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

##power method:
def power_method_stationary(P, tol=1e-9, max_iter=1000):
    """
    Finds the stationary distribution using the power method.
    
    Parameters:
        P (numpy.ndarray): The transition matrix (n x n).
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        numpy.ndarray: The stationary distribution (1 x n).
    """
    n = P.shape[0]
    # Start with an arbitrary probability distribution
    pi = np.ones(n) / n  # Uniform distribution
    
    for _ in range(max_iter):
        new_pi = pi @ P  # Update pi
        if np.linalg.norm(new_pi - pi, 1) < tol:  # Check for convergence
            return new_pi
        pi = new_pi
    
    raise ValueError("Power method did not converge within the maximum number of iterations")

# Usage:
# stationary_dist = power_method_stationary(P)
# a markov chain is reversible if pi_i*P(i,j)=pi_j*P(j,i) for all state pairs i,j
# 
def is_reversible(P, pi, tol=1e-9):
    """
    Checks if a Markov chain is reversible given its transition matrix and stationary distribution.
    
    Parameters:
        P (numpy.ndarray): Transition matrix (n x n).
        pi (numpy.ndarray): Stationary distribution (1 x n).
        tol (float): Tolerance for numerical comparison (default is 1e-9).
        
    Returns:
        bool: True if the Markov chain is reversible, False otherwise.
    """
    n = P.shape[0]
    
    # Check detailed balance for all pairs of states (i, j)
    for i in range(n):
        for j in range(n):
            # Flow balance condition: pi_i * P(i, j) == pi_j * P(j, i)
            if not np.isclose(pi[i] * P[i, j], pi[j] * P[j, i], atol=tol):
                return False  # If any pair fails, the chain is not reversible
    return True  # All pairs satisfy detailed balance