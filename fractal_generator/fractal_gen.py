""" genearate fractals """

"""
editfied by https://batchloaf.wordpress.com/2013/02/10/creating-julia-set-images-in-python/"
        and https://en.wikipedia.org/wiki/Julia_set
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

range_x = np.array([-1, 1], dtype=float)
range_y = np.array([-1, 1], dtype=float)
Nx = 1000
Ny = 1000
x = np.linspace(range_x[0], range_x[1], Nx)
y = np.linspace(range_y[0], range_y[1], Ny)

N_recursion = 100

[xx, yy] = np.meshgrid(x, y)
# c = 0.00-0.65j
# c = -0.3-0.6j
c = 0+0.65j
# c = -0.8+0.156j
r = 1


def func_recursion(z, mask):
    z_new = z*z + c
    mask = np.logical_or(mask, np.abs(z_new)>r)
    mask = np.abs(z_new)>r
    z_new[mask]=0
    return (z_new, mask)

z = xx+ yy*1j
mask = np.zeros([Ny, Nx], dtype=bool)
for i in range(N_recursion):
    [z, mask] = func_recursion(z, mask)

# z_plot = np.dstack([ np.angle(z)/(2*np.pi)+0.5, np.ones([Ny, Nx]), np.abs(z)/r ])
# z_plot = hsv_to_rgb(z_plot)
# plt.imshow(z_plot)

z_plot = np.abs(z)
plt.figure(figsize=[12,12])
plt.imshow(z_plot, cmap='gray', vmin=0, vmax=np.percentile(z_plot,98))
plt.axis('off')
# plt.colorbar()