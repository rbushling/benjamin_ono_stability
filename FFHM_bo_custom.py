import numpy as np

def FFHM(L, D, f_hats, extras=False):
    eigs = np.array([], dtype=np.complex_)
    N = len(f_hats[0,:])
    M = 3
    mid = int((N-1)/2)
    print('N = ' + str(N))
    print('mid = ' + str(mid))
    print('')
    mu_vals = []

    def frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    def augment_cols(matrix, exponent):
        for col in range(0, N):
            factor = 1j*(mu + 2*np.pi*(col-mid)/L)
            matrix[:,col] *= factor ** exponent
        return matrix

    for mu in frange(-np.pi/L, np.pi/L, 2*np.pi/(L*D)):
        f_hats_copy = np.array(f_hats, dtype=np.complex_)
        f_hats_copy[0,mid] = 1j * np.sign(mu)          # np.sign([mu + 2*np.pi*j/L for j in range(0,N)])
        if np.abs(mu) < np.pi/(2*L*D):
            f_hats_copy[0,mid] = 0

        matrix_dict = {}
        for i in range(0, M):
            mat = f_hats_copy[i,mid] * np.eye(N,dtype=np.complex_)
            for j in range(1, mid+1):
                mat += f_hats_copy[i,mid+j] * np.eye(N,k=j,dtype=np.complex_) \
                       + f_hats_copy[i,mid-j] * np.eye(N,k=-j,dtype=np.complex_)
            matrix_dict.update({M-(i+1): mat})
        matrix_dict = {key: augment_cols(value, key) for key, value in matrix_dict.items()}
        new_eigs = np.linalg.eigvals(sum(matrix_dict.values()))
        eigs = np.append(eigs, new_eigs)
        if extras:
            mu_vals = np.append(mu_vals, [mu for eig in new_eigs])

    if extras:
        return [eigs, mu_vals]
    return eigs
