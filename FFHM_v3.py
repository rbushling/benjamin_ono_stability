# Computes the spectrum of a linear differential operator using the Floquet-Fourier-Hill method.
# L is the (least) period of the operator coefficients; D is the number of Floquet exponents;
# f_hats is an array of the operator coefficients (from the highest derivative f(M-1)_hat to the
# lowest derivative f0_hat) AS A LIST OF FOURIER COEFFICIENTS [c(-n), c(-n+1), ..., c(-1), c(0),
# c(1), ..., c(n-1), c(n)]--that is, f_hats is a 1xM array of lists, each of which has length
# N=2n+1. If the boolean 'extras' is set to True, returns a pair [eigs, mu_vals] instead of just
# the eigenvalues. This is useful for plotting a 'Im(lambda) vs. mu' diagram, as in the article
# by Deconinck & Nivala. Credits to Professors Bernard Deconinck and J. Nathan Kutz for

import numpy as np

def FFHM(L, D, f_hats, extras=False):
    eigs = np.array([], dtype=np.complex_)
    N = len(f_hats[0,:])
    M = len(f_hats[:,0])
    mid = int((N-1)/2)
    mu_vals = []

    def frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    def augment_cols(matrix, exponent):
        for col in range(0, N):
            factor = 1j*mu + 2j*np.pi*(col-mid)/L
            matrix[:,col] *= factor ** exponent
        return matrix

    for mu in frange(-np.pi/L, np.pi/L, 2*np.pi/(L*D)):
        matrix_dict = {}
        for i in range(0, M):
            mat = f_hats[i,mid] * np.eye(N,dtype=np.complex_)
            for j in range(1, mid+1):
                mat += f_hats[i,mid+j] * np.eye(N,k=j,dtype=np.complex_) \
                       + f_hats[i,mid-j] * np.eye(N,k=-j,dtype=np.complex_)
            matrix_dict.update({i: mat})
        matrix_dict = {key: augment_cols(value, M-key-1) for key, value in matrix_dict.items()}
        new_eigs = np.linalg.eigvals(sum([value for value in matrix_dict.values()]))
        eigs = np.append(eigs, new_eigs)
        if extras:
            mu_vals = np.append(mu_vals, [mu for eig in new_eigs])

    if extras:
        return [eigs, mu_vals]
    return eigs
