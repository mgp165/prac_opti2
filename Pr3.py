# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:09:04 2020

@author: fran
"""

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt


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


def f(x,y,l):
    return 0.5*linalg.norm(x-y, 'fro')**2+l*tv_norm(x)


def grad_f(x,y,l):
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

# plt.imshow(I)


# ejercicio 3 y 4

l = 0.2
im = np.zeros([774, 1032, 3])
colors = ['red', 'green', 'blue']
fig, axs = plt.subplots(2)

for j in range(3):
    h = lambda x: f(x, I[:, :, j], l)
    g = lambda x: grad_f(x, I[:, :, j], l)
    
    xk = I[:, :, j]
    
    for i in range(49):
        pk = -g(xk)
        alpha = wolfe(h, g, xk, pk)
        xk_next = xk + alpha*pk
        axs[0].plot([i, i+1], [h(xk), h(xk_next)], colors[j])
        axs[1].plot([i, i+1],[linalg.norm(g(xk), 'fro'), linalg.norm(g(xk_next), 'fro')], colors[j])
        xk = xk_next
        
    im[:, :, j] = xk

# plt.imshow(im)
# plt.imsave('foto_restaurada.jpg', im)
