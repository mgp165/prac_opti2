# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:48:16 2022

@author: Manuel García Plaza
"""

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import time

# items = [1, ..., 10000]

# def power(a,b):
#     return a ** b

# num_cores = multiprocessing.cpu_count()
# itemsPostPower = Parallel(n_jobs=num_cores)(delayed(power)(item, 2) for item in items)


def tv_norm(x):
    """Computes the total variation norm. From jcjohnson/cnn-vis."""
    x_diff = x - np.roll(x, -1, axis=1)
    y_diff = x - np.roll(x, -1, axis=0)
    grad_norm2 = x_diff**2 + y_diff**2 + np.finfo(np.float32).eps
    norm = np.sum(np.sqrt(grad_norm2))
    return norm


def grad_tv_norm(x):
    """Computes the gradient of the total variation norm. From jcjohnson/cnn-vis."""
    x_diff = x - np.roll(x, -1, axis=1)
    y_diff = x - np.roll(x, -1, axis=0)
    grad_norm2 = x_diff**2 + y_diff**2 + np.finfo(np.float32).eps
    dgrad_norm = 0.5 / np.sqrt(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:] -= dx_diff[:, :-1]
    grad[1:, :] -= dy_diff[:-1, :]
    return grad


def f(x, y, l):
    return 0.5*linalg.norm(x-y, 'fro')**2+l*tv_norm(x)


def grad_f(x, y, l):
    return x-y+l*grad_tv_norm(x)


# ejercicio 1

def wolfe(f, g, x, p, c1=1e-4, c2=0.9):
    continua = True
    a = 0
    b = 2
    df_p1 = c1*np.trace(g(x).T@p)
    df_p2 = c2*np.trace(g(x).T@p)
    fx = f(x)

    while f(x + b*p) <= fx + b*df_p1:
        b = 2*b

    while continua:
        alpha = (a + b)/2
        if f(x + alpha*p) > fx + alpha*df_p1:
            b = alpha
        elif np.trace(g(x+alpha*p).T@p) < df_p2:
            a = alpha
        else:
            continua = False

    return alpha


# ejercicio 2

I = plt.imread('foto_noisy.jpg')/255
plt.figure()
plt.imshow(I)
plt.show()

# ejercicio 3 y 4

l = 0.8
im = np.zeros([774, 1032, 3])
t1 = time.time()


def reduccion_ruido(h, g):
    global xk
    pk = -g(xk)
    alpha = wolfe(h, g, xk, pk)
    xk_next = xk + alpha*pk
    xk = xk_next


for j in range(3):
    h = lambda x: f(x, I[:, :, j], l)
    g = lambda x: grad_f(x, I[:, :, j], l)
    xk = I[:, :, j]

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(reduccion_ruido)(h, g) for i in range(49))
    im[:, :, j] = xk

plt.figure()
plt.imshow(im)
plt.show()
plt.imsave('foto_restaurada2.jpg', im)
t2 = time.time()
print(str(t2-t1))
