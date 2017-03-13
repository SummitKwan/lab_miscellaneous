"""
Demo for CS2951m presentation: perceptron algorithm for large margin classifier
"""
# run ./perceptron_demo/perceptron

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')


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


def kernel_evaluate_x(x, data_keep, d=1, sigma=None):
    xs = data_keep[0]
    ys = data_keep[1]
    if sigma is None:  # polynomial kernel
        y_hat = np.sign( np.sum( ( np.sum(xs* np.expand_dims(x, 0), axis=1) +1)**d *ys) )
    else:              # rbf kernel
        y_hat = np.sign( np.sum( np.exp( -np.sum((xs-np.expand_dims(x, 0))**2, axis=1 )/(2*sigma**2) ) * ys))
    return y_hat


def kernel_perceptron_update(data_keep, x, y, d=1, sigma=None):
    xs = data_keep[0]
    ys = data_keep[1]
    if kernel_evaluate_x(x, data_keep, d, sigma) * y <=0:
        xs = np.vstack( (xs, x) )
        ys = np.append( ys, y)
    return [xs, ys]


def kernel_draw_boundary(data_keep, grid_space = np.arange(-1,1,0.01), tf_plot=True, d=1, sigma=None):

    grid_space_x, grid_space_y = np.meshgrid(grid_space, grid_space)
    N_grid = len(grid_space_x)
    x_ravel = np.zeros([N_grid**2, 2])
    x_ravel[:, 0] = grid_space_x.ravel()
    x_ravel[:, 1] = grid_space_y.ravel()
    y_ravel = [kernel_evaluate_x(x, data_keep, d, sigma) for x in x_ravel]

    grid_assign = np.array(y_ravel).reshape([N_grid, N_grid])
    if tf_plot:
        plt.contourf(grid_space, grid_space, grid_assign,
                 levels=[-1.5,0,1.5], alpha=0.1, cmap='bwr')
        plt.contour(grid_space, grid_space, grid_assign,
                    levels=[-1.5, 0, 1.5], colors=['k'])
    return grid_assign


""" original online perceptron """
np.random.seed(seed=1)

m = 500   # number of data points
dim = 2  # number of dimensions
R = 1    # largest radius of X


# generate X in Range
X = (np.random.rand(m*10, dim)*2-1)*R
X = X[np.sum(X**2, axis=1)<=R, :][:m, :]

# generate Y
v_org = np.array([0.2, 0.6, 0.8])
Y = evaluate_x(X, v_org)

if True:
    x1=X[:,0]
    x2=X[:,1]
    Y[x1>0]= 1
    Y[x1<0]=-1
    Y[x1**2+(x2+0.5)**2<=0.5**2]= 1
    Y[x1**2+(x2-0.5)**2<=0.5**2]=-1
    Y[x1**2+(x2+0.5)**2<=0.15**2]=-1
    Y[x1**2+(x2-0.5)**2<=0.15**2]=+1


plt.plot(X[Y>0,0], X[Y>0,1], 'rP')
plt.plot(X[Y<0,0], X[Y<0,1], 'bo')
draw_boundary(v_org)
plt.axis('square')
plt.xlim([-1,1])
plt.ylim([-1,1])

v=np.zeros(3)
data_keep = [np.zeros([1,2]), np.zeros(1)]
T = 5
type_perceptron = 'kernel'   # 'linear', 'kernel'
d = 5
sigma = 0.4
tf_plot = True
tf_savefig = True
i_plot_interval = m

for t in range(T):
    N_error=0
    for i in range(m):
        x = X[i, :]
        y = Y[i]

        if tf_plot and i%i_plot_interval == 0:
            plt.cla()
            if t==0:
                plt.scatter(X[:i+1, 0], X[:i+1, 1], c=Y[:i+1], cmap='bwr', vmin=-1, vmax=1, edgecolors='w')
            else:
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', vmin=-1, vmax=1, edgecolors='w')
            plt.scatter(x[0], x[1], c='g', s=100, marker='X', alpha=0.5)

            if type_perceptron =='kernel':
                kernel_draw_boundary(data_keep, d=d, sigma=sigma)
            else:
                draw_boundary(v)

            plt.axis('square')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.title('t={}/{}, n={}/{}, N_error={}/{}_before'.format(t + 1, T, i + 1, m, N_error, i + 1))

            if tf_savefig:
                plt.savefig('./perceptron_demo/temp_figs/perceptron_T{}_i{}_before.png'.format(t, i))
            else:
                plt.show()
                plt.pause(0.1)


        if type_perceptron == 'kernel':
            data_keep_new = kernel_perceptron_update(data_keep, x, y, d=d, sigma=sigma)
            if len(data_keep_new[1]) != len(data_keep[1]):
                N_error = N_error + 1
            data_keep = data_keep_new
        else:
            v_new = perceptron_update(v, x, y)
            if np.any(v_new != v):
                N_error=N_error+1
            v=v_new

        if tf_plot and i%i_plot_interval == 0:

            plt.cla()
            if t==0:
                plt.scatter(X[:i+1, 0], X[:i+1, 1], c=Y[:i+1], cmap='bwr', vmin=-1, vmax=1, edgecolors='w')
            else:
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', vmin=-1, vmax=1, edgecolors='w')
            plt.scatter(x[0], x[1], c='g', s=100, marker='X', alpha=0.5)

            if type_perceptron =='kernel':
                kernel_draw_boundary(data_keep, d=d, sigma=sigma)
            else:
                draw_boundary(v)

            plt.axis('square')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.title('t={}/{}, n={}/{}, N_error={}/{}_update'.format(t + 1, T, i + 1, m, N_error, i+1))

            if tf_savefig:
                plt.savefig('./perceptron_demo/temp_figs/perceptron_T{}_i{}_update.png'.format(t, i))
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
