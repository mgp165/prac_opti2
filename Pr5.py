# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 08:39:50 2022

@author: manug
"""

import matplotlib.pyplot as plt 
import numpy as np
from numpy import array
import scipy.optimize


# ejercicio 1

f = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


# ejercicio 2


def sum_cols(M):
    s=0
    for i in range(len(M[0])):
       s += M[:,i]
    
    return s


def NM(f, x0, it):
    n = len(x0)
    X = np.array([x0 for i in range(n+1)]).T
    
    for i in range(1,n+1):
        X[i-1,i] += 1

    fX = np.array([f(X[:,i]) for i in range(n+1)])
    orden = np.argsort(fX)
    x_min = X[:, orden[0]]
    f_min = f(x_min)
    
    for j in range(it):
       
        x_max = X[:, orden[n]]
        xb = (-x_max + sum_cols(X))/n
        
        
        f_max = f(x_max)
        x_ref = 2*xb - x_max
        f_ref = f(x_ref)
        
        if f_min > f_ref:
            x_exp = 2*x_ref - xb
            if f(x_exp) < f_ref:
                x_new = x_exp
                
            else:
                x_new = x_ref
        
            
        elif f_min <= f_ref < fX[orden[n-1]]:
            x_new = x_ref
        
        
        else:
            if f_max <= f_ref:
                x_new = (x_max + xb)/2
            else:
                x_new = (x_ref + xb)/2
        
        
        X[:, orden[n]] = x_new
        fX = np.array([f(X[:,i]) for i in range(n+1)])
        orden = np.argsort(fX)
        x_min = X[:, orden[0]]
        f_min = f(x_min)
        
    return x_min, f_min

    # hay que ver como reciclar las variables de forma correcta y eficiente


# ejercicio 3

sol = NM(f, np.array([-0.3, 3.7]), 10)

# comprobar que converge a (3,2)


# ejercicio 4

[X, Y] = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
Z = f([X, Y])


plt.figure()
plt.contour(X, Y, Z, 50)
plt.show()

# dibujar triangulos y escalar ejes


scipy.optimize.minimize(f, np.array([-0.3, 3.7]), method='Nelder-Mead')


































