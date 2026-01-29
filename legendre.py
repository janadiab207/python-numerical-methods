import numpy as np
import matplotlib.pyplot as plt

"""
Module:legendre

This module provides utilities to compute and visualize Legendre polynomials, 
which are widely used in numerical methods and mathematical physics.

Functions:
- compute_legendre: Computes Legendre polynomials up to a specified degree.
- plot_legendre: Visualizes the Legendre polynomials up to a given degree.

Testing:
The `__main__` block demonstrates the computation and visualization 
of Legendre polynomials using example inputs.

"""


def compute_legendre(x,p):

    """
    Compute Legendre polynomials up to degree p.

    This function generates Legendre polynomials for a given set of x-values
    and returns a 2D array where each row corresponds to a polynomial of a 
    specific degree.

    Parameters:
    -----------
    x : numpy array
        A 1D array of x-values where the polynomials are evaluated.
    p : int
        The maximum degree of the Legendre polynomials to compute.

    Returns:
    --------
    numpy array
        A 2D array where the rows represent the computed Legendre polynomials 
        from degree 0 to p, evaluated at the x-values.

    Example:
    --------
    >>> x = np.linspace(-1, 1, 5)
    >>> compute_legendre(x, 3)
    array([[ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ],
           [-1.   , -0.5  ,  0.   ,  0.5  ,  1.   ],
           [ 1.   ,  0.125, -0.5  ,  0.125,  1.   ],
           [-1.   , -0.4375,  0.   ,  0.4375,  1.   ]])
    """

    n_points = len(x)
    legendre = np.zeros((p + 1, n_points))
    legendre[0, :] = 1 
    
    if p >= 1:
        legendre[1, :] = x  

    for n in range(2, p + 1):
        legendre[n, :] = ((2 * n - 1) * x * legendre[n - 1, :] - (n - 1) * legendre[n - 2, :]) / n

    return legendre

def plot_legendre(p):
    """
    Plot Legendre polynomials up to a specified degree.

    This function computes and visualizes the Legendre polynomials up to the 
    specified degree p. The polynomials are plotted on the interval [-1, 1].

    Parameters:
    -----------
    p : int
        The maximum degree of Legendre polynomials to plot.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot.

    Example:
    --------
    >>> plot_legendre(3)
    (Displays a plot of Legendre polynomials up to degree 3)
    """


    #Do not alter the following line of code
    fig = plt.figure()
    
    x = np.linspace(-1, 1, 500)  
    legendre = compute_legendre(x, p) 
    for n in range(p + 1):
        plt.plot(x, legendre[n, :], label=f'L{n}(x)') 


    plt.title(f"Legendre Polynomials up to Degree {p}")
    plt.xlabel("x")
    plt.ylabel("L(x)")
    plt.legend()
    plt.grid()

    return fig

####################################################################
## Testing 
## You should test with other values, but this should be done in
## a separate file
####################################################################

if __name__ == "__main__":

    #part a
    x = np.linspace(-1,1,5)

    print("Legendre polynomials:\n")
    print(compute_legendre(x,3))

    #part b
    fig = plot_legendre(3)
    plt.show()

