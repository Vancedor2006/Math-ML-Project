# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:49:17 2026

@author: Vezin
"""

import matplotlib.pyplot as plt
import numpy as np

def cost_function(x,y):
    return x**2 + y**2

def gradient(x,y):
    return 2*x, 2*y

#gradient descent algorithm
learning_rate = 0.1
iterations = 15
#starting coordinates
current_x = 4
current_y = 3
x_history = [current_x]
y_history = [current_y]
z_history = [cost_function(current_x, current_y)]

#compute the gradient descent algorithm
for i in range (iterations):
    grad_x , grad_y = gradient(current_x, current_y)
    current_x = current_x - learning_rate * grad_x
    current_y = current_y - learning_rate * grad_y
    
    x_history.append(current_x)
    y_history.append(current_y)
    z_history.append(cost_function(current_x, current_y))


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x,y)
Z = cost_function(X, Y)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none',alpha=0.6)

ax.plot(x_history,y_history,z_history, color='red',marker = 'o', label = 'Gradient Descent Path')
ax.scatter(x_history[0],y_history[0],z_history[0],color='black',s=100, label='start')
ax.scatter(x_history[-1],y_history[-1],z_history[-1],color='black',s=100, label='minimum')

ax.set_title('3d Gradient Descent Optimization')
ax.set_xlabel('Parameter X')
ax.set_ylabel('Parameter Y')
ax.set_zlabel('Cost')
ax.legend()
fig.colorbar(surface, shrink=0.7, aspect=5)

plt.show()