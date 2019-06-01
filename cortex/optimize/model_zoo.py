# library of low-dimensional test functions for non-convex optimization
# https://www.sfu.ca/~ssurjano/optimization.html
    
import math
import numpy as np


def rosenbrock(x):    
    """
        The Rosenbrock's function of N variables
        f(x) =  100*(x_i - x_{i-1}^2)^2 + (1- x_{1-1}^2)
    """
    y = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
    # Compute the jacobian
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    j[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    j[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    j[-1] = 200*(x[-1]-x[-2]**2)
    
    # Compute the Hessian matrix
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)

    return y, j, H


def rosten(x, y, required_grad=None):
    '''
        Ed Rosten's spiral function
        
    '''
    radius2 = np.square(x) + np.square(y)
    radius = np.sqrt(radius2)
    arctan = 2 * np.arctan2(y, x);
    comb = 20 * radius + arctan;
    r = np.sqrt(np.sin(comb) * 0.5 + 0.5);
    
    if required_grad is None:
        required_grad = False
    
    if required_grad:
        comb = np.cos(comb);
        J = np.matrix([(20*x/radius - 2*y/radius2) * comb / (4 * r), (20*y/radius + 2*x/radius2) * comb / (4 * r)])
        return r, J
       
    return r
    
def rozenbrock(x, y):
    return (1-x)**2 + 100*(y - x**2)**2

# Bowl functions
def bohachevsky(x, y):
    return x**2 + 2*y**2 - 0.3*np.cos(3*3.14*x) - 0.4 * np.cos(4*3.14*y) + 0.7

def trid(x, y):
    return (x-1)**2 + (y-1)**2 - x*y

# Plate functions
def booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def matyas(x, y):
    return 0.26*(x**2 + y**2) - 0.48*x*y

def zakharov(x, y):
    return (x**2 + y**2) + (0.5*x + y)**2 + (0.5*x + y)**4

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.65 - x + x*y**3)**2

def six_hump(x, y):
    return (4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2

def beale(w, compute_grad=False):
    w1, w2 = w
    f_val = (1.5-w1+w1*w2)**2 + (2.25-w1+w1*w2**2)**2
    if compute_grad:
        grad_w1 = 2*(1.5-w1+w1*w2)*(-1+w2) + 2*(2.25-w1+w1*w2**2)*(-1+w2**2)
        grad_w2 = 2*(1.5-w1+w1*w2)*w1 + 2*(2.25-w1+w1*w2**2)*2*w1*w2
        grad = np.array([[np.asscalar(grad_w1)], 
                         [np.asscalar(grad_w2)]])
        res = (f_val, grad)
    else:
        res = f_val
    return res   