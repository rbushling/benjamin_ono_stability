# fun is a univariate, complex-valued function; modes is an integer determining the number of
# Fourier modes in the series; period is the period over which the coefficients are calculated;
# print_statements prints updates to the console when set to True (useful for functions that are
# poorly behaved under integration)
# returns a list of length 2*modes+1 of the complex Fourier coefficients in the order [-modes,
# -modes+1, ..., -1, 0, 1, ..., modes-1, modes]

import numpy as np
import importlib

cmp = importlib.import_module('complex_integrator')

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def fourier_coeffs(fun, modes, period, print_statements=False):
    output = np.zeros(2*modes+1, dtype=np.complex_)

    # speeds things up when a coefficient is, for all intents and purposes,
    # purely imaginary or real
    def truncate(n):
        if abs(np.real(output[modes+n])) < 1.0e-14:
            output[modes+n] = 1j * np.imag(output[modes+n])
        if abs(np.imag(output[modes+n])) < 1.0e-14:
            output[modes+n] = np.real(output[modes+n])

    # prints info about the progress of the computation; useful for FS that take
    # a long time to compute
    def progress(n):
        print('mode ' + str(n) + ' of ' + str(modes))
        print('c(' + str(-n) + ') = ' + str(output[modes-n]) + ', c(' + str(n) +
                  ') = ' + str(output[modes+n]))

    output[modes] = cmp.complex_quad(fun, 0.5*period) / period
    truncate(0)
    if print_statements:
        print('computing FS for ' + str(fun))
        print('mode 0 of ' + str(modes))
        print('c(0) = ' + str(output[modes]))

    # take sampling of points to determind if function is approx. real-valued
    # if so, computation can be expedited via symmetry
    sample = [fun(1.01*x) for x in frange(-0.5*period, 0.5*period, 0.05*period)]

    if sum(abs(np.imag(sample))) < 0.01 * sum(abs(np.real(sample))):
        for k in range(1, modes+1):
            output[modes-k] = cmp.complex_quad(lambda x: fun(x) *
                            np.exp(np.complex_((-2j*k*np.pi/period)*x)), 0.5*period) / period
            truncate(-k)
            output[modes+k] = np.conj(output[modes-k])
            if print_statements: progress(k)
    else:
        for k in range(1, modes+1):
            output[modes-k] = cmp.complex_quad(lambda x: fun(x) *
                            np.exp(np.complex_((-2j*k*np.pi/period)*x)), 0.5*period) / period
            output[modes+k] = cmp.complex_quad(lambda x: fun(x) *
                            np.exp(np.complex_((2j*k*np.pi/period)*x)), 0.5*period) / period
            truncate(-k)
            truncate(k)
            if print_statements: progress(k)

    return output
