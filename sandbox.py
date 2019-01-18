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

for mode in fourier_series:
    print(mode)

U = lambda z: bo.bo(z, a, k, c)
UPrime = lambda z: bo.bo_deriv(z, a, k, c)

f2_hats = np.zeros(4*N+1)
f2_hats[2*N] = -1
f1_hats = -2 * bo_four.bo_fourier_coeffs(a, k, c, 2*N)
f1_hats[2*N] -= c
f0_hats = -2 * bo_four.bo_deriv_fourier_coeffs(a, k, c, 2*N)

# Fourier coefficients for the constant solution case
# A = -0.5 * (np.sqrt(c**2 - 4*a) + c)
# g2_hats = np.zeros(2*N+1)
# g2_hats[N] = -1
# g1_hats = np.zeros(2*N+1)
# g1_hats[N] = -2 * A + c
# g0_hats = np.zeros(2*N+1)
#
# # Fourier coefficients for the zero solution case
# h2_hats = np.zeros(2*N+1)
# h2_hats[N] = -1
# h1_hats = np.zeros(2*N+1)
# h1_hats[N] = -c
# h0_hats = np.zeros(2*N+1)

f_hats = np.array([f2_hats, f1_hats, f0_hats], dtype=np.complex_)
eigs, mu_vals = hill.FFHM(2*np.pi/k, 5000, f_hats, True)

print(f_hats)

# g_hats = np.array([g2_hats, g1_hats, g0_hats], dtype=np.complex_)
# g_eigs, g_mu_vals = hill.FFHM(1, 3840, g_hats, True)

#h_hats = np.array([h2_hats, h1_hats, h0_hats], dtype=np.complex_)
#h_eigs, h_mu_vals = hill.FFHM(1, 100, h_hats, True)

plt.figure(1)
#plt.xlim([-10,10])
plt.title('Benjamin-Ono Solution and Derivative: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.plot(X, Y, label='Benjamin-Ono solution (analytic)')
plt.plot(X, [series_soln(z) for z in X], label='Benjamin-Ono solution (Fourier)')
plt.plot(X,[bo.bo_deriv(z,a,k,c) for z in X], label='Solution derivative (analytic)')
plt.plot(X,[series_deriv(z) for z in X], label='Solution derivative (Fourier)')
plt.legend()

plt.figure(2)
#plt.xlim([-100,100])
#plt.ylim([-1.e10,1.e10])
plt.title('Benjamin-Ono Stability Spectrum: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.scatter(eigs.real, eigs.imag, color='steelblue', marker='.')

plt.figure(3)
#plt.xlim([-100,100])
#plt.ylim([-50,50])
plt.title('Benjamin-Ono $Im(\lambda)$ vs. $\mu$: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.scatter(mu_vals, eigs.imag, color='mediumpurple', marker='.')
plt.grid(True)

plt.figure(4)
plt.title('Benjamin-Ono $Re(\lambda)$ vs. $\mu$: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
plt.scatter(mu_vals, eigs.real, color='gold', marker='.')
plt.grid(True)

#alpha = np.sqrt(c**2 - 4*a, dtype=np.float_)
#beta = np.sqrt(c**2 - 4*a - k**2, dtype=np.float_)

#Hu = bo_four.bo_hilbert_fourier(a, k, c, N)
#hilbert_crap = lambda z: sum([Hu[N+n] * np.exp(1j*k*n*z) for n in range(-N,N+1)])

#plt.figure(4)
#plt.title('$\mathcal{H}(u)$')
#plt.plot(X, [-beta*k*np.sin(k*z)/(alpha - beta*np.cos(k*z)) for z in X], label='a la Jeremy')
#plt.plot(X, [hilbert_crap(z) for z in X], label='a la Ryan')
#plt.legend(loc='upper right')
#plt.plot()

#plt.figure(5)
#plt.xlim([-10,10])
#plt.title('Benjamin-Ono Constant Solution: A = {}'.format(str(np.round(A,4))))
#plt.plot(X, [A for x in X], label='Benjamin-Ono constant solution')
#plt.legend()

# plt.figure(6)
# plt.title('Constant Solution Spectrum: A = {}'.format(str(np.round(A,4))))
# plt.scatter(g_eigs.real, g_eigs.imag, color=(0.8,0.1,0.2), marker='.')
#
# plt.figure(7)
# plt.title('Constant Solution $Im(\lambda)$ vs. $\mu$: A = {}'.format(str(A)))
# plt.scatter(g_mu_vals, g_eigs.imag, color=(0.1,0.8,0.6), marker='.')
# plt.grid(True)
#
# plt.figure(4)
# plt.title('Benjamin-Ono $Re(\lambda)$ vs. $\mu$: a = {}, k = {}, c = {}'.format(str(a), str(k), str(c)))
# plt.scatter(g_mu_vals, g_eigs.real, color='gold', marker='.')
# plt.grid(True)

#plt.figure(8)
#plt.xlim([-10,10])
#plt.title('Zero Solution Spectrum: c = {}'.format(str(c)))
#plt.scatter(h_eigs.real, h_eigs.imag, color=(0.8,0.1,0.2), marker='.')

#plt.figure(9)
#plt.title('Zero Solution $Im(\lambda)$ vs. $\mu$: c = {}'.format(str(c)))
#plt.scatter(h_mu_vals, h_eigs.imag, color=(0.1,0.8,0.6), marker='.')
#plt.grid(True)

plt.show()
