import numpy as np
import importlib
import matplotlib.pyplot as plt

importlib.invalidate_caches()
bo = importlib.import_module('bo_solution')
bo_four = importlib.import_module('bo_fourier_coeffs')
# note that the implementation of Hill's method being imported is CUSTOMIZED for
# this equation! i.e., it assumed that the d_xx term involves the Hilbert transform
hill = importlib.import_module('FFHM_bo_custom')

a = 3.5
k = 1.5
c = -6.5
X = np.linspace(-10, 10, 1000)
Y = [bo.bo(z,a,k,c) for z in X]

N = 25
fourier_series = bo_four.bo_fourier_coeffs(a, k, c, N)
fs_deriv = bo_four.bo_deriv_fourier_coeffs(a, k, c, N)
series_soln = lambda z: sum(fourier_series[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1))
series_deriv = lambda z: sum(fs_deriv[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1))

U = lambda z: bo.bo(z, a, k, c)
UPrime = lambda z: bo.bo_deriv(z, a, k, c)

f2_hats = np.zeros(2*N+1)
f2_hats[N] = -1
f1_hats = -2 * bo_four.bo_fourier_coeffs(a, k, c, N)
f1_hats[N] -= c
f0_hats = -2 * bo_four.bo_deriv_fourier_coeffs(a, k, c, N)

f_hats = np.array([f2_hats, f1_hats, f0_hats], dtype=np.complex_)
eigs, mu_vals = hill.FFHM(2*np.pi/k, 250, f_hats, True)

plt.figure(1)
plt.xlim([-10,10])
plt.title('Benjamin-Ono Solution and Derivative: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.plot(X, Y, label='Benjamin-Ono solution (analytic)')
plt.plot(X, [series_soln(z) for z in X], label='Benjamin-Ono solution (Fourier)')
plt.plot(X,[bo.bo_deriv(z,a,k,c) for z in X], label='Solution derivative (analytic)')
plt.plot(X,[series_deriv(z) for z in X], label='Solution derivative (Fourier)')
plt.legend()

plt.figure(2)
plt.xlim([-10,10])
plt.title('Benjamin-Ono Stability Spectrum: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.scatter(eigs.real, eigs.imag, color=(0.8,0.1,0.2), marker='.')

plt.figure(3)
plt.title('Benjamin-Ono $Im(\lambda)$ vs. $\mu$: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.scatter(mu_vals, eigs.imag, color=(0.1,0.8,0.6), marker='.')
plt.grid(True)

alpha = np.sqrt(c**2 - 4*a, dtype=np.float_)
beta = np.sqrt(c**2 - 4*a - k**2, dtype=np.float_)

Hu = bo_four.bo_hilbert_fourier(a, k, c, N)
hilbert_crap = lambda z: sum([Hu[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1)])

plt.figure(4)
plt.title('$\mathcal{H}(u)$')
plt.plot(X, [-beta*k*np.sin(k*z)/(alpha - beta*np.cos(k*z)) for z in X], label='a la Jeremy')
plt.plot(X, [hilbert_crap(z) for z in X], label='a la Ryan')
plt.legend(loc='upper right')
plt.plot()

plt.show()
