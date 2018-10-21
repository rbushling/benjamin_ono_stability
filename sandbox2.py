# This file is where "all the stuff happens." In short, the main computations currently being examined
# will be done here. After a solution's spectrum has been examined and no further computations need
# be performed, the relevant code (definition of U(y), special identities used to make the computations
# tractable, etc.) should (if necessary/appropriate) be saved in an appropriately-named file.

import importlib
import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT!!! The following line MAY prevent this program from running on some computers.
# If so, feel free to comment it out. The command ensures that the modules being imported
# are all up-to-date.
importlib.invalidate_caches()
fs = importlib.import_module('fourier_series')
hill = importlib.import_module('FFHM')
weier = importlib.import_module('weierstrass_elliptic')

# yields a range of numbers start <= i < stop, in steps of size 'step'
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

# PARAMETERS
N = 20           # num. Fourier modes
D = 300          # num. Floquet exponents
V = -1.0
#k = 0.71
E = 5.0          # in terms of Jacobi elliptic modulus, E = (k**2.-k**4.) * V**2. / (2*(2.*k**2.-1.)**2.)
C = 500.
# Weierstrass variables
g2 = (4.*V**2.)/3. - 32.*E
g3 = -8.*(V**3.)/27. - 64.*V*E/3. + 16.*C**2.
w = weier.weierstrass_elliptic(g2, g3)
e1, e2, e3 = np.complex_(w.roots)
omega1, omega3 = np.real(w.periods[0]), np.complex_(w.periods[1])
L = 2.*omega1
y0 = omega1

# returns the complex square root of a complex variable
def sq_root(x):
    return np.sqrt(x, dtype=np.complex_)

fact = lambda y: np.complex_(w.P(0.5*(y+y0)) - V/3.)
denom = lambda y: (fact(y) - 2.*sq_root(-2.*E)) * (fact(y) + 2.*sq_root(-2.*E))
PPrime = lambda y: np.complex_(w.Pprime(0.5*(y+y0)))
PPrimePrime = lambda y: np.complex_(6.*(w.P(0.5*(y+y0)))**2 - 0.5*g2)
# U and UPrime are, respectively, the solution and its derivative w.r.t. y
U = lambda y: (sq_root(2.*E)*PPrime(y) + 2.*C*fact(y)) / denom(y)
UPrime = lambda y: (denom(y) * (0.5*sq_root(2*E)*PPrimePrime(y) + C*PPrime(y)) -
                     PPrime(y)*fact(y)*(sq_root(2.*E)*PPrime(y) + 2.*C*fact(y))) / (denom(y))**2.

# OPERATOR COEFFICIENTS
f3 = lambda y: -1
f2 = lambda y: 0
f1 = lambda y: V - 6.*U(y)**2.
f0 = lambda y: -12.*U(y)*UPrime(y)

# HILL'S METHOD

f_hats = np.array([fs.fourier_coeffs(f3, N, L, True), fs.fourier_coeffs(f2, N, L, True),
                   fs.fourier_coeffs(f1, N, L, True), fs.fourier_coeffs(f0, N, L, True)])
evals, mu_vals = hill.FFHM(L, D, f_hats, True)

# PLOTS

plt.figure(1)
plt.title('Solution Spectrum: V = ' + str(round(V,3)) + ', E = ' + str(round(E,3)) + ', C = ' + str(round(C,3)))
plt.grid(True)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')

plt.figure(2)
plt.title('Im[$\lambda$] vs. $\mu$: V = ' + str(round(V,3)) + ', E = ' + str(round(E,3)) + ', C = ' + str(round(C,3)))
plt.grid(True)
plt.scatter(mu_vals, evals.imag, color=(0.8,0.05,0.4), marker='.')

# variables are cast to real below in case of small imaginary part due to numerical error
# make sure the solution being plotted actually is real-valued!!!
plt.figure(3)
plt.title("U(y) and U'(y): V = " + str(round(V,3)) + ", E = " + str(round(E,3)) + ", C = " + str(round(C,3)))
plt.grid(True)
plt.plot([x for x in frange(-L,L,0.01)], [np.real(U(x)) for x in frange(-L,L,0.01)])
plt.plot([x for x in frange(-L,L,0.01)], [np.real(UPrime(x)) for x in frange(-L,L,0.01)])

plt.show()
