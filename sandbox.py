import numpy as np
import importlib
import matplotlib.pyplot as plt

importlib.invalidate_caches()
bo = importlib.import_module('bo_solution')
bo_four = importlib.import_module('bo_fourier_coeffs')
hill = importlib.import_module('FFHM')

a = 3
k = 1
c = -4
X = np.linspace(-10, 10, 1000)
Y = [bo.bo(z,a,k,c) for z in X]

n = 25
fourier_series = bo_four.bo_fourier_coeffs(a, k, c, n)
print(fourier_series[n])
for i in range(1,n+1):
    print(fourier_series[n+i], fourier_series[n-i])
series_soln = lambda z: sum([fourier_series[n+k] * np.exp(1j*k*z) for k in range(-n,n+1)])

U = lambda z: bo.bo(z, a, k, c)
UPrime = lambda z: bo.bo_deriv(z, a, k, c)

f2_hats = bo_four.dxx_hilbert_fourier(a, k, c, n)
f1_hats = -2 * bo_four.bo_fourier_coeffs(a, k, c, n)
f1_hats[n] -= c
f0_hats = -2 * bo_four.bo_deriv_fourier_coeffs(a, k, c, n)

f_hats = np.array([f2_hats, f1_hats, f0_hats])
eigs = hill.FFHM(2*np.pi, 300, f_hats)

plt.figure(1)
plt.xlim([-10,10])
plt.plot(X,Y)
plt.plot(X,[series_soln(z) for z in X])
plt.plot(X,[bo.bo_deriv(z,a,k,c) for z in X])

plt.figure(2)
plt.scatter(eigs.real, eigs.imag, color=(0.8,0.1,0.2), marker='.')

plt.show()
