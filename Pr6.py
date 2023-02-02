# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:37:02 2022

@author: manug
"""

# np.expand_dims(v, 0 o 1)
# tomar escala logaritmica en la grafica

import numpy as np
import matplotlib.pyplot as plt

# ejercicio 1

f = lambda x: 0.5*np.array(range(len(x),0,-1))@x**2
g = lambda x: np.diag(range(len(x), 0, -1))@x
H = lambda x: np.diag(range(len(x), 0, -1))

print('Ejercicio 1:')
print('La matriz Hessiana es diagonal con todos sus elementos positivos, '\
      'desde n hasta 1, por tanto es DP y f es convexa.')

# ejercicio 2

def wolfe(f, g, x, p, c1=1e-4, c2=0.9):
    continua = True
    a = 0
    b = 2
    df_p = np.trace(g(x).T@p)
    fx = f(x)
    while f(x + b*p) <= fx + c1*b*df_p:
        b = 2*b

    while continua:
        alpha = (a + b)/2
        if f(x + alpha*p) > fx + c1*alpha*df_p:
            b = alpha
        elif np.trace(g(x+alpha*p).T@p) < c2*df_p:
            a = alpha
        else:
            continua = False
    
    return alpha


def BFGS(f, g, x0, D0, iter):
    v = np.zeros(iter+1)
    v[0] = f(x0)
    xk = np.expand_dims(x0,1)
    Dk = D0

    for i in range(iter):
        pk = - Dk@g(xk)
        alpha = wolfe(f, g, xk, pk)
        xk_next = xk + alpha*pk
        sk = xk_next - xk
        qk = g(xk_next) - g(xk)
        const1 = qk.T@Dk@qk
        const2 = Dk@qk
        const3 = sk/(sk.T@qk)
        wk = np.sqrt(const1)*(const3-const2/const1)
        Ek = const3@sk.T - (const2@const2.T)/const1 + wk@wk.T
        Dk = Dk + Ek
        xk = xk_next
        v[i+1] = f(xk)
        
    return xk, v

# ejercicio 3

x0 = np.array([i for i in range(10, 1001, 10)])
D0 = np.eye(100)
it = 125
(xit, v) = BFGS(f, g, x0, D0, it)

fig, axs = plt.subplots(3)
axs[0].semilogy([0, 1], [v[0], v[1]], 'blue')
for i in range(2,it+1):
    axs[0].semilogy([i-1, i], [v[i-1], v[i]], 'blue')
    # axs[0].title.set_text('$f(x_k)$')
    axs[1].plot([i-2, i-1], [v[i-1]/v[i-2], v[i]/v[i-1]], 'blue')
    # axs[1].title.set_text('$f(x_{k+1})/f(x_k)$')
    axs[2].plot([i-2, i-1], [v[i-1]/v[i-2]**2, v[i]/v[i-1]**2], 'blue')

print()
print('Ejercicio 3:')
print('El mínimo global de f es 0 ya que es una suma de elementos no negativos'\
      ' y únicamente se alcanza en el origen. Este resultado es justamente el '\
      'obtenido con el algoritmo y los datos iniciales dados. Se comprueba viendo'\
      ' la primera gráfica donde el valor de f va disminuyendo monótonamente '\
      '(ya que según la escala empleada, f debe tender a menos infinito cuando'\
      ' k se va a infinito).')
    
print('La sucesión de errores para evaluar la tasa de convergencia está representada'\
      ' en la segunda gráfica mediante el ratio de imágenes consecutivas ya que '\
      'el límite es 0. Vemos en esta figura que este cociente tiende a 0, por tanto,'\
      ' la convergencia es superlineal.')
    
print('De la tercera gráfica obtenemos información sobre si la convergencia es '\
      'cuadrática. Según la figura, podemos intuir una tendencia claramente '\
      'ascendente de este ratio, con lo que concluimos que la convergencia no es'\
      'cuadrática.')

# ejercicio 4

# D0 = np.ones([100, 100])
# (xit, v) = BFGS(f, g, x0, D0, it)

# fig, axs = plt.subplots(3)
# axs[0].semilogy([0, 1], [v[0], v[1]], 'black')
# for i in range(2,it+1):
#     axs[0].semilogy([i-1, i], [v[i-1], v[i]], 'black')
#     axs[1].plot([i-2, i-1], [v[i-1]/v[i-2], v[i]/v[i-1]], 'black')
#     axs[2].plot([i-2, i-1], [v[i-1]/v[i-2]**2, v[i]/v[i-1]**2], 'black')

print()
print('Ejercicio 4:')
print('El método no converge debido a que la matriz inicial no es DP, '\
      'condición exigida para los métodos quasi-Newton.')

