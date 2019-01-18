# Input:  L = period of the traveling wave solution
#         D = number of Floquet modes
#         f_hats is an np.array of size 3 x (4N+1)), where N is the number of Fourier modes used
#            in the truncated eigenvalue problem (so that the Hill matrix has size 2N+1); the
#            ith row corresponds to the (3-i)th differential operator, and the coefficients
#            in each row are ordered -2N, -2N+1, ..., -1, 0, 1, ..., 2N-1, 2N
#         extras is a boolean that should be set to True if the user wishes to receive a list
#            of mu values along with the eigenvalues
# Output: returns the Floquet-Fourier-Hill method approximation of the stability spectrum  as
#            an np.array if extras = False; otherwise, returns a two-element list containing,
#            respectively, the eigenvalues and their corresponding mu values

import numpy as np

def FFHM(L, D, f_hats, extras=False):
    eigs = np.array([], dtype=np.complex_)     # the array to be filled with the operator eigenvalues
    M = 3                                      # the number of differential operators
    N = int((len(f_hats[0,:])-1)/4)            # the number of Fourier modes used
    mid = 2*N                                  # the index of the 0th Fourier coefficient
    mu_vals = []                               # the list for mu values, if extras = True

    # a simple floating point iterator that ranges from 'start' to 'stop' in increments of 'step,'
    # where both 'star't and 'stop' are included (contrary to Python standard practice)
    def frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    # given a matrix whose diagonals are the Fourier modes of a differential operator, multiplies
    # each column by the Floquet multiplier (i*mu + 2*pi*i*k/L)^(exponent), where 'exponent' is
    # the second argument of the function, i is the imaginary unit, and k is the column number
    # relative to the middle column of the matrix, which is considered k = 0
    # returns the resultant matrix
    def augment_cols(matrix, mu, exponent):
        exponent = int(exponent)
        for col in range(0, 2*N+1):
            factor = mu + 2*np.pi*(col-N)/L
            matrix[:, col] *= (1j * factor) ** exponent
            # because exponent = 2 signals the D_xx(H(v)) operator, we also mustiply each column
            # by the Fourier symbol for the Hilbert transform in this special case
            if exponent == 2:
                matrix[:, col] *= -1j * np.sign(np.round(factor,12))
        return matrix

    # begin by computing all the 'unaugmented' matrices beforehand; we then go back and augment a copy
    # of them when we iterate over the mu values
    matrix_dict = {}     # matrices are kept in a dictionary indexed by the order of the corresponding derivative
    # iterate over each of the 3 differential operators, 0 (2nd derivative) through M-1 (0th derivative)
    for i in range(0, M):
        mat = f_hats[i, mid] * np.eye(2*N+1, dtype=np.complex_)     # 0th Fourier coefficient goes on the diagonal
        # place the jth Fourier coefficient along the -jth diagonal for 1 <= j < N-1
        # (recall that the principal diagonal is the 0th, so the last diagonal has index N-1)
        for j in range(1, 2*N+1):
            mat += f_hats[i, mid-j] * np.eye(2*N+1, k=j, dtype=np.complex_) \
                        + f_hats[i, mid+j] * np.eye(2*N+1, k=-j, dtype=np.complex_)
        matrix_dict.update({M-(i+1): mat})

    # begins Hill's method, iterating over mu values -pi/L to pi/L in increments of 2*pi/(L*D)
    for mu in frange(-np.pi/L, np.pi/L, 2*np.pi/(L*D)):
        augmented_matrix_dict = {key: augment_cols(value.copy(), mu, key) for key, value in matrix_dict.items()}

        new_eigs = np.linalg.eigvals(sum(augmented_matrix_dict.values()))
        eigs = np.append(eigs, new_eigs)
        if extras:
            mu_vals = np.append(mu_vals, [mu for i in range(0,len(new_eigs))])

    if extras:
        return [eigs, mu_vals]
    return eigs
