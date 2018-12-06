#

import numpy as np
import scipy.integrate

def test_vars(a, k, c):
    a = np.float_(a)
    k = np.float_(k)
    c = np.float_(c)
    try:
        assert k > 0 and c < 0 and k**2 < c**2 - 4*a, 'Must have k > 0, c < 0, and k**2 < c**2 - 4a; ' + \
                                        'you tried a = ' + str(a) + ', k = ' + str(k) + ', and c = ' + str(c)
    except AssertionError as error:
        print(error)
        print('Returning None')
        return [None, None, None]
    return [a, k, c]

def bo_fourier_coeffs(a, k, c, N):
    a, k, c = test_vars(a, k, c)
    if None in [a, k, c]:
        return None
    alpha = -1 * np.sqrt((c**2 - 4*a - k**2) / (c**2 - 4*a))
    factor = -k**2 / np.sqrt(c**2 - 4*a)
    coeff = factor / np.sqrt(1 - alpha**2)
    constant_term = coeff + 0.5*(np.sqrt(c**2 - 4*a) + c)
    body = (np.sqrt(1 - alpha**2) - 1) / alpha
    result = np.array([coeff * body**n for n in range(1,N+1)])
    return -np.concatenate((np.flip(np.concatenate((np.array([constant_term]), result))), result))

def bo_deriv_fourier_coeffs(a, k, c, N):
    result = bo_fourier_coeffs(a, k, c, N)
    if result is None:
        return result
    return np.array([1j * (n-N) * result[n] for n in range(0,len(result))])

def dxx_hilbert_fourier(a, k, c, N):
    result = bo_fourier_coeffs(a, k, c, N)
    if result is None:
        return result
    return np.array([1j * (n-N)**2 * np.sign(n-N) * result[n] for n in range(0,len(result))])
