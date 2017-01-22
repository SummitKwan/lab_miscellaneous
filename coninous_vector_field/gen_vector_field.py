import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def normalize(X):
    return X/(np.absolute(X)+10**(-16))
def GaussianKernel2D(sigma):
    scale = np.ceil(3*sigma)
    x = np.arange(-scale,scale+1)
    kernel1D = np.exp(-x**2/(2*sigma**2))
    kernel2D = np.expand_dims(kernel1D, axis=1) * np.expand_dims(kernel1D, axis=0)
    return kernel2D

N_grid = [64,64]
N_kernel = [11,11]
K = 2
theta_ini = (np.random.rand(*N_grid)-0.5) *(2*np.pi)
X = np.cos(theta_ini) + np.sin(theta_ini)*1j
X = normalize(X)
Y = X


Y1 = sp.signal.convolve2d(X, GaussianKernel2D(1), mode='same', boundary='wrap')
Y1 = normalize(Y1)
Y2 = sp.signal.convolve2d(X, GaussianKernel2D(3), mode='same', boundary='wrap')
Y2 = normalize(Y2)
Y3 = sp.signal.convolve2d(X, GaussianKernel2D(5), mode='same', boundary='wrap')
Y3 = normalize(Y3)




[_, h_axes] = plt.subplots(2,2)
h_axes = h_axes.flatten()
h_axes[0].matshow(np.angle(X), vmin=-np.pi, vmax=np.pi ,cmap='hsv')
# h_axes[0].quiver(np.real(X),np.imag(X))
h_axes[0].axis('off')
h_axes[1].matshow(np.angle(Y1), vmin=-np.pi, vmax=np.pi ,cmap='hsv')
# h_axes[1].quiver(np.real(Y1),np.imag(Y1))
h_axes[1].axis('off')
h_axes[2].matshow(np.angle(Y2), vmin=-np.pi, vmax=np.pi ,cmap='hsv')
# h_axes[2].quiver(np.real(Y2),np.imag(Y2))
h_axes[2].axis('off')
h_axes[3].matshow(np.angle(Y3), vmin=-np.pi, vmax=np.pi ,cmap='hsv')
# h_axes[3].quiver(np.real(Y3),np.imag(Y3))
h_axes[3].axis('off')
