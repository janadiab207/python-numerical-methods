import numpy as np
import matplotlib.pyplot as plt

"""
Module: time_steppers

This module provides numerical methods to solve ordinary differential 
equations (ODEs) using two popular techniques:

1. Explicit Euler Method (`perform_time_stepping1`): 
   A straightforward first-order method for approximating solutions.
2. Runge-Kutta 4th Order Method (`perform_time_stepping2`): 
   A higher-order, more accurate method.

Testing:
The `__main__` block demonstrates both methods by solving a simple ODE 
defined by a lambda function.

Functions:
- perform_time_stepping1: Solve ODEs using the Explicit Euler method.
- perform_time_stepping2: Solve ODEs using the Runge-Kutta 4th order method.

"""


def perform_time_stepping1(T,N,y0,y1,f):
    
    """
    Solve an ODE using the Explicit Euler method.

    The Explicit Euler method approximates the solution to an ordinary 
    differential equation (ODE) by stepping forward in time with a fixed 
    step size. It is a simple and computationally inexpensive method.

    Parameters:
    -----------
    T : float
        The total simulation time.
    N : int
        The number of time intervals (steps).
    y0 : float
        The initial condition at t=0.
    y1 : float
        The value of y at the second time step (t=1).
    f : function
        A function f(t, y) defining the derivative dy/dt.

    Returns:
    --------
    numpy array
        Array of time points (t_n).
    numpy array
        Array of solution values (y_n).

    Example:
    --------
    >>> T = 5
    >>> N = 10
    >>> y0 = 1
    >>> y1 = -1 + np.sqrt(5)
    >>> f = lambda t, y: 1 / (1 + y)
    >>> t_vals, y_vals = perform_time_stepping1(T, N, y0, y1, f)
    >>> print(t_vals, y_vals)
    """

    t_vals = np.linspace(0, T, N + 1)  
    y_vals = np.zeros(N + 1)  
    
    h = T / N 
    y_vals[0] = y0
    y_vals[1] = y1  

    
    for n in range(1, N):
        y_vals[n + 1] = y_vals[n] + h * f(t_vals[n], y_vals[n])

    return t_vals, y_vals

####################################################################

def perform_time_stepping2(T,N,M,y0,f):
    
    """
    Solve an ODE using the Runge-Kutta 4th Order Method (RK4).

    The RK4 method provides a more accurate approximation of the solution to 
    an ordinary differential equation by using intermediate slope evaluations 
    within each time step. It is one of the most widely used numerical methods 
    for solving ODEs.

    Parameters:
    -----------
    T : float
        The total simulation time.
    N : int
        The number of time intervals (steps).
    M : int
        Additional refinement parameter (not used in this implementation).
    y0 : float
        The initial condition at t=0.
    f : function
        A function f(t, y) defining the derivative dy/dt.

    Returns:
    --------
    numpy array
        Array of time points (t_n).
    numpy array
        Array of solution values (y_n).

    Example:
    --------
    >>> T = 5
    >>> N = 10
    >>> M = 5
    >>> y0 = 1
    >>> f = lambda t, y: 1 / (1 + y)
    >>> t_vals, y_vals = perform_time_stepping2(T, N, M, y0, f)
    >>> print(t_vals, y_vals)
    """

    t_vals = np.linspace(0, T, N + 1) 
    y_vals = np.zeros(N + 1)  
    
    h = T / N 
    y_vals[0] = y0

    
    for n in range(N):
        t_n = t_vals[n]
        y_n = y_vals[n]
        k1 = h * f(t_n, y_n)
        k2 = h * f(t_n + h / 2, y_n + k1 / 2)
        k3 = h * f(t_n + h / 2, y_n + k2 / 2)
        k4 = h * f(t_n + h, y_n + k3)
        y_vals[n + 1] = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_vals, y_vals

####################################################################
## Testing 
## You should test with other values, but this should be done in
## a separate file
####################################################################

if __name__ == "__main__":

    #
    T = 5
    N = 10
    M = 5
    y0 = 1
    y1 = -1+np.sqrt(5)
    f = lambda t,y: 1/(1+y)

    #part a
    tn, yn = perform_time_stepping1(T,N,y0,y1,f)

    print("perform_time_stepping1:\n")
    print("tn = ",tn)
    print("yn = ",yn)

    #part b
    tn, yn = perform_time_stepping2(T,N,M,y0,f)

    print("\nperform_time_stepping2:\n")
    print("tn = ",tn)
    print("yn = ",yn)
                                      
