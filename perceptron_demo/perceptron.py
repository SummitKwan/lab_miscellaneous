"""
Demo for Brown CS2951m presentation: Perceptron algorithm for large margin classifier
Shaobo Guan, Shaobo_Guan@brown.edu
"""
# run ./perceptron_demo/perceptron

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

""" funcitons for the parametric online perceptron algorithm """

def evaluate_x(x, v):
    """
    evalute the perceptron output
    :param x: data point, (n) vector or (m*n) array;  where n is the dim of data and m is the number of data points
    :param v: the weight parameter of perceptron. (n+1) vector, where v[0] is the bias term
    :return: binary assignment of x
    """
    if len(x.shape)==2:
        return np.sign(np.sum(x*v[1:], axis=1)+v[0])
    elif len(x.shape) == 1:
        return np.sign(np.sum(x * v[1:]) + v[0])


def voted_evaluate_x(x, V, C):
    """ for voted perceptron """
    return np.sign(np.sum(np.array([evaluate_x(x,v)*c for v,c in zip(V,C)]), axis=0))


def ini_v():
    """ generate a random data boundary """
    theta_org = np.random.rand(1) * np.pi * 2
    return np.array([(np.random.rand(1)[0] - 0.5)*0.8, np.cos(theta_org), np.sin(theta_org)])


def draw_boundary(v, grid_space = np.arange(-1,1,0.001), tf_plot=True):
    """ draw decision bounday of percetrpon with parameter v """
    grid_space_x, grid_space_y = np.meshgrid(grid_space, grid_space)
    grid_assign = np.sign(v[0] + grid_space_x*v[1] + grid_space_y*v[2])
    if tf_plot:
        plt.plot([-1, 1], [(-v[0] + v[1]) / v[2], (-v[0] - v[1]) / v[2]], 'k-')
        plt.contourf(grid_space, grid_space, grid_assign,
                 levels=[-1.5,0,1.5], alpha=0.1, cmap='bwr')
    return grid_assign


def draw_voted_boundary(V,C, grid_space = np.arange(-1,1,0.001), tf_plot_multiple_levels=False):
    """ draw decision bounday of voted percetrpon with parameter V and C """
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


""" funcitons for the non-parametric online perceptron algorithm (kernel perceptron) """


def kernel_evaluate_x(x, data_keep, d=1, sigma=None):
    """
    evalute the kernel perceptron output, can choose between polunomial kernel of degree d or RBF kernel of scale sigma,
    if sigma is None, use polynomial kernel, if sigma is a number, use RBF kernel.
    by default use polynomial kernel of degree one (identical the origian perceptron)
    :param x:         data to evaluate
    :param data_keep: the valuable data (those that the perceptron misclassified in the past) stored by the algorithm
    :param d:         degree of polynomial kernel
    :param sigma:     scale of the RBF kernel
    :return:          +1 of -1
    """
    xs = data_keep[0]
    ys = data_keep[1]
    """ calculate similarity (defined by the kernel) between the new data point with every every valuable data """
    if sigma is None:  # polynomial kernel
        y_hat = np.sign( np.sum( ( np.sum(xs* np.expand_dims(x, 0), axis=1) +1)**d *ys) )
    else:              # rbf kernel
        y_hat = np.sign( np.sum( np.exp( -np.sum((xs-np.expand_dims(x, 0))**2, axis=1 )/(2*sigma**2) ) * ys))
    return y_hat


def kernel_perceptron_update(data_keep, x, y, d=1, sigma=None):
    """
    update the kernel perceptron: if misclassified, append the x, y to data_keep,
    which stores all the valuable data points and their labels
    """
    xs = data_keep[0]
    ys = data_keep[1]
    if kernel_evaluate_x(x, data_keep, d, sigma) * y <=0:
        xs = np.vstack( (xs, x) )
        ys = np.append( ys, y)
    return [xs, ys]


def kernel_draw_boundary(data_keep, grid_space = np.arange(-1,1,0.01), tf_plot=True, d=1, sigma=None):
    """ draw decision bounday of kernel perceptron """
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




""" ====================  online perceptron  ==================== """

""" <<<<<<<<<< parameters to choose for users, start ---------- """
data_boundary = 'linear'     # the true boundary of data, choose from ['linear', 'curved', 'tai-chi']

T = 1                        # number of iterations over the dataset
type_perceptron = 'parametric'   # choose from ['parametic', 'kernel']
d = 5                        # degree of polynomial kernel
sigma = 0.2                  # scale of RBF kernel, if None, use polynomial kernel, otherwise if a number, use rbf kernel
tf_plot = True               # if plot
tf_savefig = False           # if save figures
i_plot_interval = 1          # the interval of plotting, eg., if set to 10, plot once after going over every 10 data points
path_savefig =  './perceptron_demo/temp_figs'   # path to save figures

if type_perceptron == 'kernel':
    print('use the kernel perceptron')
    if sigma is None:
        print('kernel: polynomial kernel with degree {}'.format(d))
    else:
        print('kernel: RBF kernel with scale {}'.format(sigma))
else:
    print('use the origianl parametric perceptron')

""" ---------- parameters to choose for users, end   >>>>>>>>>> """


""" generate data  """
np.random.seed(seed=1)

m = 50   # number of data points
dim = 2   # number of dimensions
R = 1     # largest radius of X


# generate X in Range
X = (np.random.rand(m*10, dim)*2-1)*R
X = X[np.sum(X**2, axis=1)<=R, :][:m, :]

# assigne labels to x

if data_boundary == 'linear':
    v_org = np.array([0.2, 0.6, 0.8])
    Y = evaluate_x(X, v_org)
elif data_boundary == 'curved':
    x1=X[:,0]
    x2=X[:,1]
    Y[x1>0]= 1
    Y[x1<0]=-1
    Y[x1**2+(x2+0.5)**2<=0.5**2]= 1
    Y[x1**2+(x2-0.5)**2<=0.5**2]=-1
elif data_boundary == 'tai-chi':
    x1=X[:,0]
    x2=X[:,1]
    Y[x1>0]= 1
    Y[x1<0]=-1
    Y[x1**2+(x2+0.5)**2<=0.5**2]= 1
    Y[x1**2+(x2-0.5)**2<=0.5**2]=-1
    Y[x1**2+(x2+0.5)**2<=0.15**2]=-1
    Y[x1**2+(x2-0.5)**2<=0.15**2]=+1
else:
    print('parameter data_boundary not recognized, must be one of "linear", "curved" or "tai-chi" ')


""" plot the origianl data """
plt.plot(X[Y>0,0], X[Y>0,1], 'rP')
plt.plot(X[Y<0,0], X[Y<0,1], 'bo')
plt.axis('square')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.title('all data points to be classified')
print('the original data')
plt.pause(1.0)


""" initial parameters for the online perceptron """
v=np.zeros(3)                        # for the orignal parametric perceptron
data_keep = [X[0:1,:], Y[0:1]]       # for the kernel perceptron

print('perceptron online updating starts')
for t in range(T):
    N_error=0
    for i in range(m):
        x = X[i, :]
        y = Y[i]

        """ plot before update """
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
                plt.savefig('{}/perceptron_T{}_i{}_before.png'.format(path_savefig, t, i))
            else:
                plt.show()
                plt.pause(0.1)

        """ perceptron update step """
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

        """ plot after update """
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
                plt.savefig('{}/perceptron_T{}_i{}_update.png'.format(path_savefig, t, i))
            else:
                plt.show()
                plt.pause(0.1)


