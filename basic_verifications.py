# some code used to verify the basic functionality and correctness of the main modules

import numpy as np
import importlib
import matplotlib.pyplot as plt

importlib.invalidate_caches()
bo = importlib.import_module('bo_solution')
bo_four = importlib.import_module('bo_fourier_coeffs')
# note that the implementation of Hill's method being imported is CUSTOMIZED for
# this equation! i.e., it assumed that the d_xx term involves the Hilbert transform
hill = importlib.import_module('FFHM_bo_custom')

a = -1.0
k = 2.5
c = -4.5
X = np.linspace(-10, 10, 1000)
Y = [bo.bo(z,a,k,c) for z in X]

N = 25
fourier_series = bo_four.bo_fourier_coeffs(a, k, c, N)
fs_deriv = bo_four.bo_deriv_fourier_coeffs(a, k, c, N)
series_soln = lambda z: sum(fourier_series[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1))
series_deriv = lambda z: sum(fs_deriv[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1))

U = lambda z: bo.bo(z, a, k, c)
UPrime = lambda z: bo.bo_deriv(z, a, k, c)

# Fourier coefficients for the constant solution case
A = 0.5 * (np.sqrt(c**2 - 4*a) + c)
g2_hats = np.zeros(4*N+1)
g2_hats[2*N] = -1
g1_hats = np.zeros(4*N+1)
g1_hats[2*N] = -2 * A + c
g0_hats = np.zeros(4*N+1)

# Fourier coefficients for the zero solution case
h2_hats = np.zeros(4*N+1)
h2_hats[2*N] = -1
h1_hats = np.zeros(4*N+1)
h1_hats[2*N] = c
h0_hats = np.zeros(4*N+1)

g_hats = np.array([g2_hats, g1_hats, g0_hats], dtype=np.complex_)
g_eigs, g_mu_vals = hill.FFHM(1, 300, g_hats, True)

h_hats = np.array([h2_hats, h1_hats, h0_hats], dtype=np.complex_)
h_eigs, h_mu_vals = hill.FFHM(1, 300, h_hats, True)

alpha = np.sqrt(c**2 - 4*a, dtype=np.float_)
beta = np.sqrt(c**2 - 4*a - k**2, dtype=np.float_)

Hu = bo_four.bo_hilbert_fourier(a, k, c, N)
hilbert_transform_of_bo_fourier = lambda z: sum([Hu[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1)])

plt.figure(1)
plt.title('$\mathcal{H}(u)$')
plt.plot(X, [-beta*k*np.sin(k*z)/(alpha - beta*np.cos(k*z)) for z in X], label='a la Jeremy')
plt.plot(X, [hilbert_transform_of_bo_fourier(z) for z in X], label='a la Ryan')
plt.legend(loc='upper right')
plt.plot()

plt.figure(2)
plt.title('Benjamin-Ono Constant Solution: A = {}'.format(str(np.round(A,4))))
plt.plot(X, [A for x in X], label='Benjamin-Ono constant solution')
plt.legend()

plt.figure(3)
plt.title('Constant Solution Spectrum: A = {}'.format(str(np.round(A,4))))
plt.scatter(g_eigs.real, g_eigs.imag, color=(0.8,0.1,0.2), marker='.')

plt.figure(4)
plt.title('Constant Solution $Im(\lambda)$ vs. $\mu$: A = {}'.format(str(A)))
plt.scatter(g_mu_vals, g_eigs.imag, color=(0.1,0.8,0.6), marker='.')
plt.grid(True)

plt.figure(5)
plt.title('Constant Solution $Re(\lambda)$ vs. $\mu$: A = {}'.format(str(A)))
plt.scatter(g_mu_vals, g_eigs.real, color='gold', marker='.')
plt.grid(True)

plt.figure(6)
plt.title('Zero Solution Spectrum: c = {}'.format(str(c)))
plt.scatter(h_eigs.real, h_eigs.imag, color=(0.8,0.1,0.2), marker='.')

plt.figure(7)
plt.title('Zero Solution $Im(\lambda)$ vs. $\mu$: c = {}'.format(str(c)))
plt.scatter(h_mu_vals, h_eigs.imag, color=(0.1,0.8,0.6), marker='.')
plt.grid(True)

plt.show()
