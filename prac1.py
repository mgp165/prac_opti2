import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import minimize


#ejercicio 1

#Los puntos críticos de f son (0,0), (2,2) y (4,4)

f = lambda x: 9*x[0]**2 - 2*x[0]*x[1] + x[1]**2 - 4*x[0]**3 + 1/2*x[0]**4

g = lambda x: np.array([18*x[0] - 2*x[1] - 12*x[0]**2 + 2*x[0]**3, -2*x[0] + 2*x[1]])

H = lambda x: np.array([[18-24*x[0]+6*x[0]**2, -2], [-2, 2]])

O = np.array([0, 0])
autoval = np.linalg.eigvals(H(O))
print(autoval)
M = autoval[0]
m = autoval[1]

#como los dos autovalores son positivos, la matriz es DP por tanto se verifica la condición suficiente de ser mínimo


#ejercicio 2

[X, Y] = np.meshgrid(np.linspace(-1, 5), np.linspace(-3, 8))
Z = f([X, Y])

fig1 = plt.figure()
ejes = plt.axes(projection='3d')
ejes.plot_surface(X, Y, Z)
plt.show()


#ejercicio 3

fig2 = plt.figure()
plt.contour(X, Y, Z, 50)
plt.show()


#ejercicio 4

alpha = 2/(m + M)
x0 = np.array([0.4, -1.9])


def met_grad(grad, semilla, paso, it):
    lista = [semilla]
    for i in range(it):
        lista.append(lista[i] - paso*grad(lista[i]))

    return lista


x_k_descensorapido = met_grad(g, x0, alpha, 40)
print(f(x_k_descensorapido[40]))


#ejercicio 5


def met_newtonpuro(grad, B, semilla, it):
    lista = [semilla]
    for i in range(it):
        p = np.linalg.solve(B(lista[i]), grad(lista[i]))
        lista.append(lista[i] - p)

    return lista


x_k_newtonpuro = met_newtonpuro(g, H, x0, 5)
print(f(x_k_newtonpuro[5]))

#esto converge mucho más rápido que el anterior


#ejercicio 6


def met_busqueda(f, grad, semilla, it):
    lista = [semilla]
    for i in range(it):
        phi = lambda alpha: f(lista[i] - alpha*grad(lista[i]))
        alpha = minimize(phi, 1)
        lista.append(lista[i] - alpha.x*grad(lista[i]))

    return lista


x_k_busqlienal = met_busqueda(f, g, x0, 40)
print(f(x_k_busqlienal[40]))

#peor que newton y parecido al primero


#ejercicio 7

p_d = - g(x0)/np.linalg.norm(g(x0))
inv = np.linalg.inv(H(x0))
p_n = - np.dot(inv, g(x0))/np.linalg.norm(np.dot(inv, g(x0)))

phi_d = lambda a: f(x0+a*p_d)
phi_n = lambda a: f(x0+a*p_n)

I = np.linspace(0, 2)

fig3 = plt.figure()
plt.plot(I, f([x0[0] + I*p_d[0], x0[1] + I*p_d[1]]))
plt.plot(I, f([x0[0] + I*p_n[0], x0[1] + I*p_n[1]]))
plt.legend(['p_d', 'p_n'])
plt.show()

#viendo todo el intervalo es preferible p_n porque alcanza el menor valor peroen pasos pequeños es mejor p_d


#ejercicio 8 (opcional)

[X2, Y2] = np.meshgrid(np.linspace(-1, 1), np.linspace(-2, 1))
Z2 = f([X2, Y2])

x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []

fig4 = plt.figure()
plt.contour(X2, Y2, Z2, 50)

for i in range(len(x_k_descensorapido)):
    x1.append(x_k_descensorapido[i][0])
    y1.append(x_k_descensorapido[i][1])
    plt.plot(x1[i], y1[i], 'ro')
plt.plot(x1, y1, 'red')

for i in range(len(x_k_newtonpuro)):
    x2.append(x_k_newtonpuro[i][0])
    y2.append(x_k_newtonpuro[i][1])
    plt.plot(x2[i], y2[i], 'bo')
plt.plot(x2, y2, 'blue')

for i in range(len(x_k_busqlienal)):
    x3.append(x_k_busqlienal[i][0])
    y3.append(x_k_busqlienal[i][1])
    plt.plot(x3[i], y3[i], 'go')
plt.plot(x3, y3, 'green')
plt.show()

u = np.diff(x3)
v = np.diff(y3)

es_ortogonal = True

for i in range(1, len(u)):
    pe = np.dot([u[i-1], v[i-1]], [u[i], v[i]])
    if pe > 10**-6:
        es_ortogonal = False
        break

if es_ortogonal:
    print('Todos los segmentos son ortogonales. ')
