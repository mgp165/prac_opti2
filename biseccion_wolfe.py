import numpy as np

'''
Proposición 1.7
Algoritmo de bisección para las condiciones de Wolfe.
Se elige 0<c1<c2<1. Se toma inicialmente a = 0, b = 2, continua = True.
Típicamente c1 = 10^-4 y c2 = 0.9 si Newton o c2=0.1 si Gradiente Conjugado.
Hipótesis:
    f de clase C1
    p dirección de descenso de f en x
    f acotada inferiormente en la semirrecta {x+alpha*p / alpha>0}
'''
a = 0
b = 2
c1 = 10**-4
c2 = 0.9
continua = True

f = lambda x: x[0]**2 + 3*x[1]
grad = lambda x: np.array([2*x[0], 3])
x0 = np.array([1, 3])
p1 = np.array([1, 1])
p2 = p1/np.linalg.norm(p1)
f_p = lambda x, p: np.dot(grad(x0), p2)

while f(x0 + b*p2) <= f(x0) + c1*b*f_p(x0, p2):
    b = 2*b

while continua:
    alpha = (a + b)/2
    if f(x0 + alpha*p2) > f(x0) + c1*alpha*f_p(x0, p2):
        b = alpha
    elif f_p(x0 + alpha*p2, p2) < c2*f_p(x0, p2):
        a = alpha
    else:
        continua = False

#tenemos alpha verificando Armijo y curvatura.
