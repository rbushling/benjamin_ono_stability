#

import numpy as np

def bo_fourier_coeffs(a, k, c, n): # CURRENTLY ASSUMING k = 1
    a = np.float_(a)
    k = np.float_(k)
    c = np.float_(c)
    try:
        assert c < 0 and k**2 < c**2 - 4*a
    except AssertionError as error:
        print(error)
        print('Returning None')
        return None
    alpha = -1 / np.sqrt((c**2 - 4*a) / (c**2 - 4*a - k**2))
    factor = k**2 / (np.sqrt(c**2 - 4*a - k**2) * np.sqrt((c**2 - 4*a) / (c**2 - 4*a - k**2)))
    coeff = factor / np.sqrt(1 - alpha**2)
    body = (np.sqrt(1 - alpha**2) - 1) / alpha
    result =  np.array([coeff * body**k for k in range(1,n+1)])
    return np.concatenate((np.flip(np.concatenate((np.array([2*coeff]),result))), result))
