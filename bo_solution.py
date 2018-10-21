# Evaluates the traveling wave solution to the Benjamin-Ono equation, the NEGATIVE of Bronski/Hur/Johnson
# equation 5.10, at the point z with parameters a, k, and c. Prints an error and returns None if the
# conditions c < 0 and k**2 < c**2 - 4a are not both satisfied.

import numpy as np

def test_vars(z, a, k, c):
    z = np.float_(z)
    a = np.float_(a)
    k = np.float_(k)
    c = np.float_(c)
    try:
        assert c < 0 and k**2 < c**2 - 4*a, 'Must have c < 0 and k**2 < c**2 - 4a; you tried a = ' + \
                                            str(a) + ', k = ' + str(k) + ', and c = ' + str(c)
    except AssertionError as error:
        print(error)
        print('Returning None')
        return [None, None, None, None]
    return [z, a, k, c]

def bo(z, a, k, c):
    z, a, k, c = test_vars(z, a, k, c)
    if None in [z, a, k, c]:
        return None
    numerator = -k**2 / np.sqrt(c**2 - 4*a - k**2)
    denominator = np.sqrt((c**2 - 4*a) / (c**2 - 4*a - k**2)) - np.cos(k*z)
    return numerator / denominator + 0.5 * (np.sqrt(c**2 - 4*a) + c)

def bo_deriv(z, a, k, c):
    z, a, k, c = test_vars(z, a, k, c)
    if None in [z, a, k, c]:
        return None
    numerator = k**2 / np.sqrt(c**2 - 4*a - k**2)
    denominator = (np.sqrt((c**2 - 4*a) / (c**2 - 4*a - k**2)) - np.cos(k*z))**2
    return numerator * k * np.sin(k*z) / denominator
