# Computes the spectrum of a linear differential operator using the Floquet-Fourier-Hill method.
# L is the (least) period of the operator coefficients; D is the number of Floquet exponents;
# f_hats is an array of the operator coefficients (from the highest derivative f3_hat to the
# lowest derivative f0_hat) AS A LIST OF FOURIER COEFFICIENTS [c(-n), c(-n+1), ..., c(-1), c(0),
# c(1), ..., c(n-1), c(n)]--that is, f_hats is a 1x4 array of lists, each of which has length
# 2n+1. The boolean 'extras' returns a pair [evals, mu_vals] instead of just the eigenvalues
# when set to True. This is useful for plotting a 'Im(lambda) vs. mu' diagram, as in the article
# by Deconinck & Nivala.

import numpy as np

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def FFHM(L,D,f_hats,extras=False):
    evals = np.array([], dtype=np.complex_)
    N = len(f_hats[0,:])
    mid = int((N-1)/2)
    mu_vals = []
    for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
        f3_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[0,mid] for q in range(0,N)])
        f2_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[1,mid] for q in range(0,N)])
        f1_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[2,mid] for q in range(0,N)])
        f0_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[3,mid] for q in range(0,N)])
        for n in range(1,mid+1):
            f3_matrix += np.diag([f_hats[0,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[0,mid-n] for q in range(0,N-n)], -n)
            f2_matrix += np.diag([f_hats[1,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[1,mid-n] for q in range(0,N-n)], -n)
            f1_matrix += np.diag([f_hats[2,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[2,mid-n] for q in range(0,N-n)], -n)
            f0_matrix += np.diag([f_hats[3,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[3,mid-n] for q in range(0,N-n)], -n)
        for m in range(0,N):
            fact = (1j*mu + 2j*np.pi*(m-mid)/L)
            f3_matrix[:,m] = [f3_matrix[k,m] * fact**3 for k in range(0,N)]
            f2_matrix[:,m] = [f2_matrix[k,m] * fact**2 for k in range(0,N)]
            f1_matrix[:,m] = [f1_matrix[k,m] * fact for k in range(0,N)]
        new_evals = np.linalg.eigvals(f3_matrix + f2_matrix + f1_matrix + f0_matrix)
        evals = np.append(evals, new_evals)
        if extras:
            mu_vals = np.append(mu_vals, [mu for eig in new_evals])
    if extras:
        return [evals, mu_vals]
    return evals
