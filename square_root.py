import numpy as np
import math

"""
Module: square_root

In this modulle we compute the approximate square root of any positive number by applying
iterative Taylor series expansion method,which enables for accuracy analysis of the approximation
and expermentation with different taylor items.

Functions:
- approx_square_root: Approximates the square root using Taylor series.
- compute_relative_error: Computes the relative error between the approximation 
  and the actual square root.
- experiment_relative_error: Runs experiments to test the accuracy of the 
  approximation for different Taylor series configurations.

  This module can be used to try different numerical methods for square root computation and 
study the behavior of iterative methods with varying levels of precision.
"""

def taylor_term(k, delta_x, x_n):
    if k == 0:
        return 1
    coeff = (-1)**k / ((1 - 2*k) * 4**k)
    binomial = math.comb(2*k, k)
    return coeff * binomial * (delta_x / x_n**2)**k


def approx_square_root(a, N, x0, eps):

    """
    Approximate the square root of a positive number using Taylor series.

    Parameters:
    -----------
    a : float
        The number to compute the square root of (a > 0).
    N : int
        The number of terms in the Taylor expansion.
    x0 : float
        The initial guess for the square root.
    eps : float
        The error tolerance for stopping the iterative method.

    Returns:
    --------
    float
        The computed approximate square root of a.
    int
        The number of iterations required.

    Example:
    --------
    >>> approx_square_root(16, 5, 3, 1e-6)
    (4.0, 3)
    """

    x_new = x0
    iteration_count = 0

    while abs(x_new**2 - a) > eps:  
        delta_x = a - x_new**2 
        taylor_sum = sum(taylor_term(k, delta_x, x_new) for k in range(N + 1))
        x_new = x_new * taylor_sum
        iteration_count += 1

    return x_new, iteration_count

####################################################################
## Testing 
## You should test with other values, but this should be done in
## a separate file
####################################################################

if __name__ == "__main__":

    #Test 1
    value, iteration_count = approx_square_root(25,2,3,1.0e-13)
    print("sqrt(25) = ",value,", No_iterations = ",iteration_count)

    #Test 2
    value, iteration_count = approx_square_root(10,3,3,1.0e-13)
    print("sqrt(10) = ",value,", No_iterations = ",iteration_count)



