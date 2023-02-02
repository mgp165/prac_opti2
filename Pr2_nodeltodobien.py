# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:10:30 2022

@author: Manuel García Plaza and Javier Beviá Ripoll
"""

import numpy as np
import matplotlib.pyplot as plt

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
plt.title('Ejercicio 1')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()

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

fig3 = plt.figure()
plt.contour(X, Y, Z, curvas_nivel)

x0 = np.array([-0.1, -0.4])
n = 501
lista_dr = np.zeros([n, 2])
lista_dr[0] = x0

plt.axis([-0.5, 1.1, -0.5, 1.1])
plt.plot(lista_dr[0][0], lista_dr[0][1], 'ko', markersize=2,linewidth=None,markerfacecolor='black')

for i in range(1, n):
    xk_dr = lista_dr[i-1]
    pk_dr = -g(xk_dr)
    alpha = backtracking(f, g, xk_dr, pk_dr)
    lista_dr[i] = xk_dr + alpha*pk_dr
    plt.plot(lista_dr[i][0], lista_dr[i][1], 'ko', markersize=2,linewidth=None,markerfacecolor='black')
    plt.plot([lista_dr[i-1][0], lista_dr[i][0]], [lista_dr[i-1][1], lista_dr[i][1]], 'g-')
plt.title('Ejercicio 3') 
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()

print('f(x0) = ' + str(f(x0)))
print('x500 = ' + str(lista_dr[-1]))
print('f(x500) = ' + str(f(lista_dr[-1])))


# ejercicio 4

fig4 = plt.figure()
plt.contour(X, Y, Z, curvas_nivel)

lista_n = np.zeros([6, 2])
lista_n[0] = x0

plt.axis([-0.5, 1.1, -0.5, 1.1])
plt.plot(lista_n[0][0], lista_n[0][1], 'ko', markersize=3,linewidth=None,markerfacecolor='black')

for i in range(1, 6):
    xk_n = lista_n[i-1]
    pk_n = np.linalg.solve(H(xk_n), g(xk_n))
    lista_n[i] = xk_n - pk_n
    plt.plot(lista_n[i][0], lista_n[i][1], 'ko', markersize=3,linewidth=None,markerfacecolor='black')
    plt.plot([lista_n[i-1][0], lista_n[i][0]], [lista_n[i-1][1], lista_n[i][1]], 'r-')
plt.title('Ejercicio 4')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.savefig('Pr2.png',bbox_inches='tight',dpi=300)
plt.show()

print('x5 = ' + str(lista_n[-1]))
print('f(x5) = ' + str(f(lista_n[-1])))


# ejercicio 5

condH = np.linalg.cond(H(x_opt))
print('El número de condición del Hessiano de f en (1,1) es ' + str(condH))

# debido al elevadísimo número de condición de esta matriz, la convergencia es muy lenta


# ejercicio 6 (opcional)

# pasamos al problema equivalente 1/2 norma al cuadrado
# es una función de una variable que desarrollando la expresión es una parábola convexa
# entonces el mínimo (global y único) se alcanza en el vértice: alpha*=-b/(2a)

fig6 = plt.figure()
plt.contour(X, Y, Z, curvas_nivel)

m = 46
alpha0 = backtracking(f, g, x0, -g(x0))
lista_op = np.zeros([m, 2])
lista_op[0] = x0
lista_op[1] = x0 - alpha0*g(x0)

plt.axis([-0.5, 1.1, -0.5, 1.1])
plt.plot(lista_op[0][0], lista_op[0][1], 'ko', markersize=3,linewidth=None,markerfacecolor='black')
plt.plot(lista_op[1][0], lista_op[1][1], 'ko', markersize=3,linewidth=None,markerfacecolor='black')
plt.plot([lista_op[0][0], lista_op[1][0]], [lista_op[0][1], lista_op[1][1]], 'b-')

for i in range(2, m):
    xk_op1 = lista_op[i-1]
    xk_op2 = lista_op[i-2]
    pk_op = -g(xk_op1)
    alphak = (g(xk_op1)-g(xk_op2))@(xk_op1-xk_op2)/(np.linalg.norm(g(xk_op1)-g(xk_op2))**2)
    lista_op[i] = xk_op1 + alphak*pk_op
    plt.plot(lista_op[i][0], lista_op[i][1], 'ko', markersize=3,linewidth=None,markerfacecolor='black')
    plt.plot([lista_op[i-1][0], lista_op[i][0]], [lista_op[i-1][1], lista_op[i][1]], 'b-')
plt.title('Ejercicio 6')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()

print('x45 = ' + str(lista_op[-1]))
print('f(x45) = ' + str(f(lista_op[-1])))

# la convergencia de este método es sustancialmente mejor que la del descenso más rápido
# pero un tanto más lenta que la del método de Newton puro
