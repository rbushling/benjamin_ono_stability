import numpy as np
import importlib
import matplotlib.pyplot as plt

importlib.invalidate_caches()
bo = importlib.import_module('bo_solution')
bo_four = importlib.import_module('bo_fourier_coeffs')

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

plt.figure()
plt.xlim([-10,10])
plt.plot(X,Y)
plt.plot(X,[series_soln(z) for z in X])
plt.show()
