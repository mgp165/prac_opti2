# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:10:30 2022

@author: Manuel García Plaza and Javier Beviá Ripoll
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# función de Rosenbrock, su gradiente y su Hessiano
f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
g = lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
H = lambda x: np.array([[2 - 400*(x[1]-x[0]**2) + 800*x[0]**2, -400*x[0]], [-400*x[0], 200]])

# el único punto crítico y mínimo global
x_opt = np.array([1, 1]) 


# ejercicio 1

[X, Y] = np.meshgrid(np.linspace(-2.5, 2.5), np.linspace(-3, 6))
Z = f([X, Y])
curvas_nivel = [i for i in range(0, 501, 20)]

fig1 = plt.figure()
plt.contour(X, Y, Z, curvas_nivel)
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Práctica 2.')

t = 0.5
x_1 = np.array([1, 1])
y_1 = np.array([-1, 1])

if f(x_1*t + y_1*(1-t)) > t*f(x_1) + (1-t)*f(y_1):
    print('f no es convexa ya que los puntos (1,1) y (-1,1) con t = 0.5 no cumplen la definición.')


# ejercicio 2

def backtracking(f, g, xk, pk, alpha=1, rho=0.5, c=1e-4):
    cte1 = f(xk)
    cte2 = c*g(xk)@pk
    while f(xk + alpha*pk) > cte1  + alpha*cte2:
        alpha = rho*alpha
    return alpha


# ejercicio 3

x0 = np.array([-0.1, -0.4])

plt.axis([-0.5, 1.1, -0.5, 1.1])

xk = x0

for i in range(500):
    pk = -g(xk)
    alpha = backtracking(f, g, xk, pk)
    xk_next = xk + alpha*pk
    plt.plot([xk[0], xk_next[0]], [xk[1], xk_next[1]], 'g-', marker='o',\
             markerfacecolor='black', markeredgecolor='black',  markersize=2)
    xk = xk_next
   
print('f(x0) = ' + str(f(x0)))
print('x500 = ' + str(xk_next))
print('f(x500) = ' + str(f(xk_next)))


# ejercicio 4

xk = x0

for i in range(5):
    pk = np.linalg.solve(H(xk), g(xk))
    xk_next = xk - pk
    plt.plot([xk[0], xk_next[0]], [xk[1], xk_next[1]], 'r-', marker='o',\
             markerfacecolor='black', markeredgecolor='black',  markersize=2)
    xk = xk_next

print('x5 = ' + str(xk_next))
print('f(x5) = ' + str(f(xk_next)))


# ejercicio 5

# Sabemos que la matriz hessiana es simétrica (cumple las condiciones del teorema de 
# Schwarz). Comprobaremos si es DP mediante el cálculo de sus valores propios, 
# hemos leído que calcular la descomposición de Cholesky mediante np.linalg.cholesky
# es más eficiente que el cálculo de los valores propios para esta comprobación
# ya que si la matriz no es DP dará error bajo el supuesto de simetría ya que 
# el comando no comprueba si la matriz es simétrica), pero puesto que usaremos
# uno de estos valores propios para calcular el número de condición creemos
# que, si bien más computacionalmente costoso, es una manera sensata de hacerlo.
# Aunque para comprobar que es DP no es lo más eficiente, para calcular el 
# número de condición nos viene bien tener ordenada la lista de autovalores

eig = np.sort(np.linalg.eigvalsh(H(x_opt))) #ordenamos la lista de valores propios

if eig[0]>0: #si el primero es positivo (puesto que está ordenada de menor a mayor) todos serán positivos 
    print("La matriz es DP.")

# Tenemos que en estas condiciones el número de condición es el cociente entre
# el valor propio más grande y el más pequeño
condH = eig[-1]/eig[0]
print('El número de condición del Hessiano de f en (1,1) es ' + str(condH))

# debido al elevadísimo número de condición de esta matriz, la convergencia es muy lenta
# por lo visto en la Proposición 1.14 de la teoría.


# ejercicio 6 (opcional)

# Pasamos al problema equivalente 1/2 norma al cuadrado
# Es una función de una variable que desarrollando la expresión es una parábola convexa
# entonces el mínimo (global y único) se alcanza en el vértice: alpha*=-b/(2a)

alpha0 = backtracking(f, g, x0, -g(x0))
xk = x0
xk_next = xk - alpha0*g(xk)

plt.plot([xk[0], xk_next[0]], [xk[1], xk_next[1]], 'b-', marker='o',\
         markerfacecolor='black', markeredgecolor='black',  markersize=2)


for i in range(44):
    pk = -g(xk_next)
    alphak = (g(xk_next)-g(xk))@(xk_next-xk)/(np.linalg.norm(g(xk_next)-g(xk))**2)
    xk = xk_next
    xk_next = xk_next + alphak*pk
    plt.plot([xk[0], xk_next[0]], [xk[1], xk_next[1]], 'b-', marker='o',\
             markerfacecolor='black', markeredgecolor='black',  markersize=2)


print('x45 = ' + str(xk_next))
print('f(x45) = ' + str(f(xk_next)))

# La convergencia de este método es sustancialmente mejor que la del descenso más rápido
# pero un tanto más lenta que la del método de Newton puro

colors = ['green', 'red', 'blue']
lines = [Line2D([0], [0], color=c) for c in colors]
labels = ['Descenso más rápido', 'Newton', 'Quasi-Newton']
plt.legend(lines, labels)
plt.savefig('Pr2.png',bbox_inches='tight',dpi=300)
