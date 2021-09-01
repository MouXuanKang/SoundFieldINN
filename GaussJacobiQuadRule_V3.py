# -*- coding: utf-8 -*-
"""
@Title:
    Gauss Quadrature Rules

@author: 
    Ehsan Kharazmi
    Division of Applied Mathematics
    Brown University
    ehsan_kharazmi@brown.edu

Created on Fri Apr 12 15:06:19 2019
"""

import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi


##################################################################
# Recursive generation of the Jacobi polynomial of order n
# For fixed \(\alpha, \beta\), the polynomials \(P_n^{(\alpha, \beta)}\)
# are orthogonal over \([-1, 1]\) with weight function \((1 - x)^\alpha(1 + x)^\beta\).
def Jacobi(n, a, b, x):
    x = np.array(x)
    return jacobi(n, a, b)(x)


##################################################################
# Derivative of the Jacobi polynomials
def DJacobi(n, a, b, x, k: int):
    x = np.array(x)
    # Gamma
    # Parameters:
    # zarray_like, Real or complex valued argument
    # Returns:
    # scalar or ndarray, Values of the gamma function
    ctemp = gamma(a+b+n+1+k)/(2**k)/gamma(a+b+n+1)
    return ctemp*Jacobi(n-k, a+k, b+k, x)

    
##################################################################
# Weight coefficients
def GaussJacobiWeights(Q: int, a, b):
    # Compute the sample points and weights for Gauss-Jacobi quadrature.
    # The sample points are the roots of the nth degree Jacobi polynomial,
    # w(x)=(1-x)^alpha*(1+x)^beta
    # Parameters:
    # Q: int,   quadrature order
    # a: float, alpha must > -1
    # b: float, beta  must > -1
    [X, W] = roots_jacobi(Q, a, b)
    # return: X:sample points; W:Weights
    return [X, W]


##################################################################
# Weight coefficients
def GaussLobattoJacobiWeights(Q: int, a, b):
    # Parameters:
    # Q: int,   quadrature order
    # a: float, alpha must > -1
    # b: float, beta  must > -1
    # return: X:sample points; W:Weights
    W = []
    X = roots_jacobi(Q-2, a+1, b+1)[0]
    if a == 0 and b == 0:
        W = 2/((Q-1) * Q * (Jacobi(Q-1, 0, 0, X) ** 2))
        Wl = 2/((Q-1) * Q * (Jacobi(Q-1, 0, 0, -1) ** 2))
        Wr = 2/((Q-1) * Q * (Jacobi(Q-1, 0, 0, 1) ** 2))
    else:
        W = 2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/((Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1, a, b, X)**2))
        Wl = (b+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/((Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1, a, b, -1)**2))
        Wr = (a+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/((Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1, a, b, 1)**2))
    W = np.append(W, Wr)
    W = np.append(Wl, W)
    X = np.append(X, 1)
    X = np.append(-1, X)
    return [X, W]
##################################################################

