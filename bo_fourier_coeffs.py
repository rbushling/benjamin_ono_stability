# "If you've imported numpy as np, you're doing scientific computing."

import numpy as np

# Converts the variables to floats and guarantees their values are in the required ranges.
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

# Returns a length 2N+1 array of the Fourier series solution coefficients (in complex)
# exponential form) to the Benjamin-Ono equation.
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
    # Uses the evenness of the function to generate the terms -N through -1 by symmetry.
    return np.concatenate((np.flip(np.concatenate((np.array([constant_term]), result))), result))

# Returns a length 2N+1 array of the Fourier series solution derivative coefficients.
def bo_deriv_fourier_coeffs(a, k, c, N):
    result = bo_fourier_coeffs(a, k, c, N)
    if result is None:
        return result
    return np.array([1j * k * (n-N) * result[n] for n in range(0,len(result))])

# Returns a length 2N+1 array of the Hilbert-transformed Fourier series solution coefficients.
def bo_hilbert_fourier(a, k, c, N):
    result = bo_fourier_coeffs(a, k, c, N)
    if result is None:
        return result
    return np.array([-1j * np.sign(n-N) * result[n] for n in range(0,len(result))])
