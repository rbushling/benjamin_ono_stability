import numpy as np
#import scipy.signal.hilbert as hilbert
import importlib
import matplotlib.pyplot as plt

importlib.invalidate_caches()
bo = importlib.import_module('bo_solution')
bo_four = importlib.import_module('bo_fourier_coeffs')
#hill = importlib.import_module('FFHM')
#fs = importlib.import_module('fourier_series')

a = 3
k = 1
c = -4
X = np.linspace(-10, 10, 1000)
Y = [bo.bo(z,a,k,c) for z in X]

#def hilbert_transform(z):
#    return np.imag(hilbert(z))

n = 25
fourier_series = bo_four.bo_fourier_coeffs(a, k, c, n)
print(fourier_series[n])
for i in range(1,n+1):
    print(fourier_series[n+i], fourier_series[n-i])
series_soln = lambda z: sum([fourier_series[n+k] * np.exp(1j*k*z) for k in range(-n,n+1)])

U = lambda z: bo.bo(z, a, k, c)
UPrime = lambda z: bo.bo_deriv(z, a, k, c)

#f3 = lambda z: 0
#f2 = lambda z: -1 * hilbert_transform(z)
#f1 = lambda z: -k - 2 * U(z)
#f0 = lambda z: -2 * UPrime(z)

#f_hats = np.array([fs.fourier_coeffs(f3, n, 2*np.pi), fs.fourier_coeffs(f2, n, 2*np.pi),
#                   fs.fourier_coeffs(f1, n, 2*np.pi), fs.fourier_coeffs(f0, n, 2*np.pi)])

#eigs = hill.FFHM(2*np.pi, 300, f_hats)

plt.figure(1)
plt.xlim([-10,10])
plt.plot(X,Y)
plt.plot(X,[series_soln(z) for z in X])
plt.plot(X,[bo.bo_deriv(z,a,k,c) for z in X])
plt.show()
