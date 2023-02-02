# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:32:43 2022

@author: Lucía Navarro y Lucía Bonet
"""


import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from matplotlib.lines import Line2D

#Definimos la función Rosenbrock, su gradiente y matriz hessiana:
f = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2 # función Rosenbrock
g = lambda x: np.array([400*(x[0]**3 - x[0]*x[1]) +2*(x[0] - 1), 200*(x[1] - x[0]**2)]) # gradiente
H = lambda x: np.array([[400*(3*x[0]**2 -x[1]) + 2, -400*x[0]], [-400*x[0] , 200]]) # hessiana

############### 1
#Curvas de nivel en el intervalo [-2.5,2.5]x[-3,6] con los valores de la función de 0 a 500 con incrementos de 20
[X,Y] = np.meshgrid(np.linspace(-2.5,2.5), np.linspace(-3,6)) 
Z = f([X,Y])
fig = plt.figure(figsize = (8, 8))
curvas = plt.contour(X,Y,Z,np.array(range(0,501,20)))
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

############### 2
#La función backtracking devuelve un valor de alpha que verifica la condición de Armijo en x_k
#Condición de Armijo: f(x_k + α*p_k) ≤ f(x_k) + c*α*(∇f_k)T*p_k con c en (0,1)
def backtracking(f,g,xk,pk,alpha=1,rho=0.5,c=10**(-4)):
    while f(xk+alpha*pk) > f(xk) + c*alpha*np.dot(g(xk).T,pk):
        alpha=rho*alpha
    return alpha

############### 3
#Método de descenso más rápido con backtracking

def mdesback(x0,i):
    lista=[x0] #añadimos las iteraciones en esta lista
    for j in range(i):
        ##para cada iteración calculamos el alpha obtenido al aplicar backtracking
        alpha = backtracking(f,g,lista[j],-g(lista[j]),alpha=1,rho=0.5,c=10**(-4)) 
        lista.append(lista[j] - alpha*g(lista[j])) #añadimos la j-ésima iteración dada por el m. de descenso más rápido
        
        plt.plot(np.array([lista[j][0],lista[j+1][0]]),np.array([lista[j][1],lista[j+1][1]]),'g',label="Backtracking") #segmentos que unen iteraciones consecutivas
        plt.plot(lista[j][0],lista[j][1],'o',color='black') #gráfica con las iteraciones
        plt.axis([-0.2,1.1,-0.5,1.1])
    return lista

plt.xlabel('eje x')
plt.ylabel('eje y')

plt.plot()

x0=np.array([0.1,0.6])
print('Ejercicio 3')
print('La imagen de x0=(0.1,0.6) es', f(x0))
print('La 500-ésima iteración es', mdesback(x0,500)[500] ,'y su imagen es', f(mdesback(x0,500)[500]))




############### 4
#Método de Newton puro (alpha=1)
def mnewp(x0, i):
    lista = [x0] #añadimos las iteraciones en esta lista
    for j in range(i): 
        lista.append(lista[j] - np.dot(np.linalg.inv(H(lista[j])), g(lista[j]))) #añadimos la iteración j-ésima dada por el m. de Newton puro
        plt.plot(np.array([lista[j][0],lista[j+1][0]]), np.array([lista[j][1],lista[j+1][1]]),'red',label="Newton puro")
        plt.plot(lista[j][0],lista[j][1],'o',color='black') #gráfica con las iteraciones
    return lista

plt.xlabel('eje x')
plt.ylabel('eje y')



plt.savefig('Pr2')
print('Ejercicio 4')
print('La 500-ésima iteración es', mnewp(x0,5)[5],'y su imagen es',f(mnewp(x0,5)[5]))


############### 5
#Basándote en el valor de la matriz hessiana en el punto (1, 1), explica en un comentario a qué 
#puede deberse la convergencia lenta del método de descenso más rápido

#Vimos que el método de descenso más rápido converge lentamente cuando el número 
#de condición de H es grande. 
#Calcularemos el número de condición y para ello estudiamos si H es DP y simétrica
print('Ejercicio 5')
print('La matrix hessiana en (1,1),', H(np.array([1,1])), ',es simétrica') 
print('Los autovalores de H,', np.linalg.eigvals(H(np.array([1,1]))), ', son ambos positivos no nulos')

autovalores = np.sort(np.linalg.eigvals(H(np.array([1,1]))))
m = autovalores[0] #menor autovalor
M = autovalores[1] #mayor autovalor
print('Como el número de condición,', M/m, ',es mucho mayor que 1 el método de descenso converge lentamente')




############ 6
print("Ejercicio 6")

def qnewton(f,g,x0,i):
    a0=backtracking(f, g, x0, -g(x0))
    lista=[x0]
    alpha=[a0]
    for j in range(i):
        lista.append(lista[j]-alpha[j]*g(lista[j]))
        alpha.append(  ( (g(lista[j+1])[0]-g(lista[j])[0])*(lista[j][0]-lista[j+1][0])+(g(lista[j+1])[1]-g(lista[j])[1])*(lista[j][1]-lista[j+1][1])   )  /( (g(lista[j+1])[0]-g(lista[j])[0])**2  + (g(lista[j+1])[1]-g(lista[j])[1])**2  ))
        plt.plot(np.array([lista[j][0],lista[j+1][0]]), np.array([lista[j][1],lista[j+1][1]]),'blue',label="quasi-Newton")
        plt.plot(lista[j][0],lista[j][1],'o',color='black') #gráfica con las iteraciones
    return lista

colors = ['green', 'red', 'blue']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
labels = ['Backtracking', 'Pure Newton', 'quasi-Newton']
plt.legend(lines, labels)

qnewton(f,g,x0,45)
print("Las iteraciones del método quasi-Newton son:",qnewton(f,g,x0,45))
        
        
        
        
        
    