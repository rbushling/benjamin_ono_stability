#

import numpy as np

def test_vars(a, k, c):
    a = np.float_(a)
    k = np.float_(k)
    c = np.float_(c)
    try:
        assert c < 0 and k**2 < c**2 - 4*a, 'Must have c < 0 and k**2 < c**2 - 4a; you tried a = ' + \
                                            str(a) + ', k = ' + str(k) + ', and c = ' + str(c)
    except AssertionError as error:
        print(error)
        print('Returning None')
        return [None, None, None]
    return [a, k, c]

def bo_fourier_coeffs(a, k, c, n): # CURRENTLY ASSUMING k = 1
    a, k, c = test_vars(a, k, c)
    if None in [a, k, c]:
        return None
    alpha = -1 / np.sqrt((c**2 - 4*a) / (c**2 - 4*a - k**2))
    factor = -k**2 / (np.sqrt(c**2 - 4*a - k**2) * np.sqrt((c**2 - 4*a) / (c**2 - 4*a - k**2)))
    coeff = factor / np.sqrt(1 - alpha**2)
    body = (np.sqrt(1 - alpha**2) - 1) / alpha
    result =  np.array([coeff * body**k for k in range(1,n+1)])
    return np.concatenate((np.flip(np.concatenate((np.array([2*coeff]),result))), result))

def bo_deriv_fourier_coeffs(a, k, c, n):
    result = bo_fourier_coeffs(a, k, c, n)
    if result is None:
        return result
    return np.array([1j * (k-n) * result[k] for k in range(0,len(result))])

def dxx_hilbert_fourier(a, k, c, n):
    result = bo_fourier_coeffs(a, k, c, n)
    if result is None:
        return result
    return np.array([1j * (k-n)**2 * np.sign(k-n) * result[k] for k in range(0,len(result))])
