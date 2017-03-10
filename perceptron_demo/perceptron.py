"""
Demo for CS2951m presentation: perceptron algorithm for large margin classifier
"""
# run ./perceptron_demo/perceptron

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

m = 50   # number of data points
dim = 2  # number of dimensions
X = np.random.rand(m, dim)*2-1

def evaluate_x(x, v):
    if len(x.shape)==2:
        return np.sign(np.sum(x*v[1:], axis=1)+v[0])
    elif len(x.shape) == 1:
        return np.sign(np.sum(x * v[1:]) + v[0])

def voted_evaluate_x(x, V, C):
    return np.sign(np.sum(np.array([evaluate_x(x,v)*c for v,c in zip(V,C)]), axis=0))

def ini_v():
    theta_org = np.random.rand(1) * np.pi * 2
    return np.array([(np.random.rand(1)[0] - 0.5)*0.8, np.cos(theta_org), np.sin(theta_org)])

def draw_boundary(v, grid_space = np.arange(-1,1,0.001), tf_plot=True):
    grid_space_x, grid_space_y = np.meshgrid(grid_space, grid_space)
    grid_assign = np.sign(v[0] + grid_space_x*v[1] + grid_space_y*v[2])
    if tf_plot:
        plt.plot([-1, 1], [(-v[0] + v[1]) / v[2], (-v[0] - v[1]) / v[2]], 'k-')
        plt.contourf(grid_space, grid_space, grid_assign,
                 levels=[-1.5,0,1.5], alpha=0.1, cmap='bwr')
    return grid_assign

def draw_voted_boundary(V,C, grid_space = np.arange(-1,1,0.001), tf_plot_multiple_levels=False):
    grid_space_x, grid_space_y = np.meshgrid(grid_space, grid_space)
    grid_assign = grid_space_x*0
    for v,c in zip(V,C):
        grid_assign += draw_boundary(v, grid_space, tf_plot=False)*c
    grid_assign_bin = np.sign(grid_assign)
    if tf_plot_multiple_levels:
        plt.contourf(grid_space, grid_space, grid_assign,
                     alpha=0.5, cmap='bwr')
    else:
        plt.contourf(grid_space, grid_space, grid_assign_bin,
                    levels=[-1.5,0,1.5], alpha=0.1, cmap='bwr')
    plt.contour(grid_space, grid_space, grid_assign_bin,
                    levels=[-1.5, 0, 1.5], colors=['k'])

def perceptron_update(v, x, y):
    if evaluate_x(x, v) * y <=0:
        v = v + y*np.array([1.0, x[0], x[1]])
    return v



""" original perceptron """

v_org = ini_v()
Y = evaluate_x(X, v_org)

plt.plot(X[Y>0,0], X[Y>0,1], 'rP')
plt.plot(X[Y<0,0], X[Y<0,1], 'bo')
draw_boundary(v_org)
plt.axis('square')
plt.xlim([-1,1])
plt.ylim([-1,1])

v=np.zeros(3)
T = 5
tf_plot = False
tf_savefig = False
for t in range(T):
    N_error=0
    for i in range(m):

        x=X[i,:]
        y=Y[i]
        v_new = perceptron_update(v, x, y)
        if np.any(v_new != v):
            N_error=N_error+1
        v=v_new

        if tf_plot:
            plt.cla()
            plt.scatter(X[:,0], X[:,1], c=Y, cmap='bwr', vmin=-1, vmax=1,edgecolors='w')
            plt.scatter(x[0], x[1], c='g', s=100, marker='X', alpha=0.5)

            draw_boundary(v)

            plt.axis('square')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.title('t={}/{}, n={}/{}, N_error={}/{}'.format(t + 1, T, i + 1, m, N_error, i+1))

            if tf_savefig:
                plt.savefig('./perceptron_demo/temp_figs/perceptron_T{}_i{}.png'.format(t, i))
            else:
                plt.show()
                plt.pause(0.1)


""" voted perceptron """
v=np.zeros(3)
T = 10
tf_plot = True
tf_savefig = True
V = [v]
C = [0]
for t in range(T):
    N_error=0
    for i in range(m):

        x=X[i,:]
        y=Y[i]
        v_new = perceptron_update(v, x, y)
        if np.any(v_new != v):
            N_error=N_error+1
            V.append(v_new)
            C.append(1)
        else:
            C[-1] += 1
        v=v_new

        if tf_plot:
            plt.cla()
            plt.scatter(X[:,0], X[:,1], c=Y, cmap='bwr', vmin=-1, vmax=1,edgecolors='w')
            plt.scatter(x[0], x[1], c='g', s=100, marker='X', alpha=0.5)

            draw_boundary(v)

            plt.axis('square')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.title('t={}/{}, n={}/{}, N_error={}/{}'.format(t + 1, T, i + 1, m, N_error, i+1))

            if tf_savefig:
                plt.savefig('./perceptron_demo/temp_figs/perceptron_T{}_i{}.png'.format(t, i))
            else:
                plt.show()
                plt.pause(0.1)

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', vmin=-1, vmax=1, edgecolors='w')
plt.scatter(x[0], x[1], c='g', s=100, marker='X', alpha=0.5)
draw_voted_boundary(V,C, tf_plot_multiple_levels=True)
plt.axis('square')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
