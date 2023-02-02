# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:48:38 2022

@author: Manuel García Plaza
"""
import numpy as np
from numpy import exp, cos, pi, sin, array


exponencial = lambda x: exp(x[1]+x[2]**2-x[3])

g1 = lambda x: x[0]**2 + x[1]*x[2]*x[3] - 2
g2 = lambda x: x[0] + x[0]*x[1] - x[2]*x[3] - 1
g3 = lambda x: -x[0] + x[0]**2 + 2*x[1] - x[2]**3 + x[3]**2 - 2
g4 = lambda x: x[0]*exponencial(x) - exp(1)
g5 = lambda x: x[1]*cos(pi*x[0]) - x[2]*sin(pi*x[3]) + 1

g = lambda x: array([g1(x), g2(x), g3(x), g4(x), g5(x)])


# ejercicio 1

f = lambda x: 0.5*(g1(x)**2 + g2(x)**2 + g3(x)**2 + g4(x)**2 + g5(x)**2)

grad_g1 = lambda x: array([2*x[0], x[2]*x[3], x[1]*x[3], x[1]*x[2]])
grad_g2 = lambda x: array([1 + x[1], x[0], -x[3], -x[2]])
grad_g3 = lambda x: array([-1 + 2*x[0], 2, -3*x[2]**2, 2*x[3]])
grad_g4 = lambda x: array([exponencial(x), x[0]*exponencial(x), 2*x[0]*x[2]*exponencial(x), -x[0]*exponencial(x)])
grad_g5 = lambda x: array([-pi*x[1]*sin(pi*x[0]), cos(pi*x[0]), -sin(pi*x[3]), -pi*x[2]*cos(pi*x[3])])

grad_g = lambda x: array([grad_g1(x), grad_g2(x), grad_g3(x), grad_g4(x), grad_g5(x)]).T
grad_f = lambda x: grad_g(x)@g(x)


# ejercicio 2

x0 = array([0, 0.2, 0.7, 0.5])
I = np.eye(4)

xk = x0
for i in range(1000):
    grad_f_xk = grad_f(xk)
    if np.linalg.norm(grad_f_xk) > 1e-7:
        grad_g_xk = grad_g(xk)
        xk = xk - np.linalg.solve(grad_g_xk @ grad_g_xk.T, I) @ grad_f_xk
    else:
        break

print('Ejercicio 2')
print('El número de iteraciones es', i+1)
print('La solución es', xk)
print('f(xk) =', f(xk))
print('norma(grad_f(xk)) =', np.linalg.norm(grad_f_xk))
print()


# ejercicio 3


def wolfe(f, g, x, p, c1=1e-4, c2=0.9):
    continua = True
    a = 0
    b = 2
    df_p = g(x).T@p
    cte1 = c1*df_p
    cte2 = c2*df_p
    fx = f(x)
    while f(x + b*p) <= fx + cte1*b:
        b = 2*b

    while continua:
        alpha = (a + b)/2
        if f(x + alpha*p) > fx + cte1*alpha:
            b = alpha
        elif g(x+alpha*p).T@p < cte2:
            a = alpha
        else:
            continua = False
    
    return alpha

xk = x0
for i in range(1000):
    grad_f_xk = grad_f(xk)
    if np.linalg.norm(grad_f_xk) > 1e-7:
        grad_g_xk = grad_g(xk)
        pk = - np.linalg.solve(grad_g_xk @ grad_g_xk.T, I) @ grad_f_xk
        alpha = wolfe(f, grad_f, xk, pk)
        xk = xk + alpha*pk
    else:
        break

print('Ejercicio 3')
print('El número de iteraciones es', i+1)
print('La solución es', xk)
print('f(xk) =', f(xk))
print('norma(grad_f(xk)) =', np.linalg.norm(grad_f_xk))
print()

# El método empleado en este ejercicio converge en un menor número de iteraciones
# a la solución que el del ejercicio anterior


# ejercicio 4

xk = x0
for i in range(1000):
    grad_f_xk = grad_f(xk)
    if np.linalg.norm(grad_f_xk) > 1e-7:
        pk = - grad_f_xk
        alpha = wolfe(f, grad_f, xk, pk)
        xk = xk + alpha*pk
    else:
        break

print('Ejercicio 4')
print('El número de iteraciones es', i+1)
print('La solución es', xk)
print('f(xk) =', f(xk))
print('norma(grad_f(xk)) =', np.linalg.norm(grad_f_xk))
print()


# ejercicio 5

x1 = array([-0.3, 1.5, -0.2, 0.8])

# experimento ejercicio 2

xk = x1
for i in range(1000):
    grad_f_xk = grad_f(xk)
    if np.linalg.norm(grad_f_xk) > 1e-7:
        grad_g_xk = grad_g(xk)
        xk = xk - np.linalg.solve(grad_g_xk @ grad_g_xk.T, I) @ grad_f_xk
    else:
        break
    
print('Ejercicio 5')
print('El número de iteraciones es', i+1)
print('La solución es', xk)
print('f(xk) =', f(xk))
print('norma(grad_f(xk)) =', np.linalg.norm(grad_f_xk))
print()

# experimento ejercicio 3

xk = x1
for i in range(1000):
    grad_f_xk = grad_f(xk)
    if np.linalg.norm(grad_f_xk) > 1e-7:
        grad_g_xk = grad_g(xk)
        pk = - np.linalg.solve(grad_g_xk @ grad_g_xk.T, I) @ grad_f_xk
        alpha = wolfe(f, grad_f, xk, pk)
        xk = xk + alpha*pk
    else:
        break

print('El número de iteraciones es', i+1)
print('La solución es', xk)
print('f(xk) =', f(xk))
print('norma(grad_f(xk)) =', np.linalg.norm(grad_f_xk))
print()

# El método del ejercicio 3 (con búsqueda lineal satisfaciendo Wolfe)
# obtiene una solución mucho mejor que el método Gauss-Newton puro (no converge en 1000 iteraciones).


# ejercicio 6 (opcional)

# Tomando como punto semilla el origen no podemos aplicar los algoritmos anteriores
# porque están definidos para matrices no singulares (y en este punto no se da esta condición)
# y por tanto no podemos aplicar el método.

x2 = np.zeros(4)

xk = x2
for i in range(1000):
    grad_f_xk = grad_f(xk)
    m = np.linalg.norm(grad_f_xk)**2
    D = m*I
    if np.linalg.norm(grad_f_xk) > 1e-7:
        grad_g_xk = grad_g(xk)
        pk = - np.linalg.solve(grad_g_xk @ grad_g_xk.T + D, I) @ grad_f_xk
        alpha = wolfe(f, grad_f, xk, pk)
        xk = xk + alpha*pk
    else:
        break

print('Ejercicio 6')
print('El número de iteraciones es', i+1)
print('La solución es', xk)
print('f(xk) =', f(xk))
print('norma(grad_f(xk)) =', np.linalg.norm(grad_f_xk))
