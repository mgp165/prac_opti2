# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:44:05 2022

@author: Bonet
"""
from numpy import array, trace
from scipy import linalg
import numpy as np
from math import e
import matplotlib.pyplot as plt

g1 = lambda x: x[0]**2+ x[1]*x[2]*x[3] -2
grad_g1 = lambda x: np.array([2*x[0], x[2]*x[3], x[1]*x[3], x[1]*x[2]])

g2 = lambda x: x[0] + x[0]*x[1] - x[2]*x[3] -1
grad_g2 = lambda x: np.array([1+x[1], x[0], -x[3], -x[2]])

g3 = lambda x: -x[0] + x[0]**2 + 2*x[1] -x[2]**3 + x[3]**2  -2
grad_g3 = lambda x: np.array([-1+2*x[0], 2, -3*x[2]**2, 2*x[3]])

g4 = lambda x: x[0]*np.exp((x[1]+x[2]**2 -x[3])) - e
grad_g4 = lambda x: np.array([np.exp((x[1]+x[2]**2 -x[3])), x[0]*np.exp((x[1]+x[2]**2 -x[3])), x[0]*2*x[2]*np.exp((x[1]+x[2]**2 -x[3])),-x[0]*np.exp((x[1]+x[2]**2 -x[3]))])

g5 = lambda x: x[1]*np.cos(np.pi*x[0]) - x[2]*np.sin(np.pi*x[3]) +1
grad_g5 = lambda x: np.array([-x[0]*np.pi*np.sin(np.pi*x[0]), np.cos(np.pi*x[0]), -np.sin(np.pi*x[3]), -x[2]*np.pi*np.cos(np.pi*x[3])])

sum_gi = lambda x: g1(x)**2 + g2(x)**2 + g3(x)**2 + g4(x)**2 + g5(x)**2 

######### EJERCICIO 1
f = lambda x: sum_gi(x)/2
grad_f = lambda x: np.dot(g1(x),np.transpose(grad_g1(x))) + np.dot(g2(x),np.transpose(grad_g2(x))) + np.dot(g3(x),np.transpose(grad_g3(x))) + np.dot(g4(x),np.transpose(grad_g4(x))) + np.dot(g5(x),np.transpose(grad_g5(x)))


######### EJERCICIO 2
print('EJERCICIO 2')
def g(x0):
    return np.array([g1(x0),g2(x0),g3(x0),g4(x0),g5(x0)])

def dg(x0):
    return np.transpose(np.array([grad_g1(x0),grad_g2(x0),grad_g3(x0),grad_g4(x0),grad_g5(x0)]))

x0 = np.array([0,0.2,0.7,0.5])
def GNpuro(x0):
    lista=[x0]
    for i in range(1000):
       if np.linalg.norm(grad_f(lista[i])) > 10**(-7):
            lista.append(lista[i]-np.dot(linalg.inv(np.dot(dg(lista[i]),np.transpose(dg(lista[i])))),np.dot(dg(lista[i]),g(lista[i]))))
       else:
           break
    return lista[-1],len(lista)
        
print('El número de iteraciones es', GNpuro(x0)[1])
print('La solución obtenida es', GNpuro(x0)[0],'con valor',g(GNpuro(x0)[0])) 
print('La norma del gradiente es',np.linalg.norm(grad_f(GNpuro(x0)[0])))   

######### EJERCICIO 3

