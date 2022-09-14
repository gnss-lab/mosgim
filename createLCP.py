import numpy as np
import lemkelcp as lcp
import scipy.special as sp
import gc
from scipy.sparse import lil_matrix, csr_matrix, issparse
import matplotlib.pyplot as plt

nbig = 15  # max order of spherical harmonic expansion
mbig = 15  # max degree of spherical harmonic expansion (0 <= mbig <= nbig)
nT = 24  # number of time steps
n_coefs = (nbig + 1)**2 - (nbig - mbig) * (nbig - mbig + 1)


def calc_coefs(M, N, theta, phi):
    """
    :param M: meshgrid of harmonics degrees
    :param N: meshgrid of harmonics orders
    :param theta: LT of IPP in rad
    :param phi: co latitude of IPP in rad
    :param sf: slant factor
    """
    n_coefs = len(M)
    a = np.zeros(n_coefs)
    Ymn = sp.sph_harm(np.abs(M), N, theta, phi)  # complex harmonics on meshgrid
    #  introducing real basis according to scipy normalization
    a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
    a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
    a[M == 0] = Ymn[M == 0].real
    del Ymn
    return a
vcoefs = np.vectorize(calc_coefs, excluded=['M','N'], otypes=[np.ndarray])


def construct(nbig, mbig, theta, phi, timeindex):
    """
    :param nbig: maximum order of spherical harmonic
    :param mbig: maximum degree of spherical harmonic
    :param theta: array of LTs of IPPs in rads
    :param phi: array of co latitudes of IPPs in rads
    """
 
    # Construct matrix of the problem (A)
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sp.sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
 

    len_rhs = len(phi)

    a = vcoefs(M=M, N=N, theta=theta, phi=phi)

    print('coefs done', n_coefs)

    del theta
    del phi
    del M
    del N
    gc.collect()

    #prepare (A) in csr sparse format
    data = np.empty(len_rhs * n_coefs)
    rowi = np.empty(len_rhs * n_coefs)
    coli = np.empty(len_rhs * n_coefs)

    for i in range(0, len_rhs,1): 
        data[i * n_coefs: (i + 1) * n_coefs] = a[i]
        rowi[i * n_coefs: (i + 1) * n_coefs] = i * np.ones(n_coefs).astype('int32')
        
        coli[i * n_coefs: (i + 1) * n_coefs] = np.arange(timeindex[i] * n_coefs, (timeindex[i] + 1) * n_coefs, 1).astype('int32')

    
    A = csr_matrix((data, (rowi, coli)), shape=(len_rhs, (nT + 1) * n_coefs))
    print('matrix (A) done')

    return A







colat = np.arange(2.5, 180, 2.5)
mlt = np.arange(0., 365., 5.)
mlt_m, colat_m  = np.meshgrid(mlt, colat)

mlt_m = np.tile(mlt_m.flatten(), nT + 1)
colat_m = np.tile(colat_m.flatten(), nT + 1)
time_m = np.array([int(_ / (len(colat) * len(mlt))) for _ in range(len(colat) * len(mlt) * (nT + 1))])



G = construct(nbig, mbig, np.deg2rad(mlt_m), np.deg2rad(colat_m), time_m)


inputfile = 'res_data_rel_gm_2017_002.npz'
outputfile = 'res_data_rel_gm_2017_002_lcp.npz'


# load data
data = np.load(inputfile, allow_pickle=True)

c0 = data['res']
N = data['N']
del data
gc.collect()



Ninv = np.linalg.inv(N)
w = G.dot(c0)
idx = (w<0)


Gnew = G[idx,:]
wnew = Gnew.dot(c0)

print 'constructing M'

NGT = Ninv * Gnew.transpose()
M = Gnew.dot(NGT)





sol = lcp.lemkelcp(M,wnew,10000)


c = c0 + NGT.dot(sol[0])
w = G.dot(c)

#levels=np.arange(0,50,0.5)
#for i in np.arange(0, nT + 1, 1):
#    plt.contourf(w[i * len(colat)*len(mlt): (i + 1) * len(colat) * len(mlt)].reshape(len(colat), len(mlt)), levels)
#    plt.colorbar()    
#    plt.show()


np.savez(outputfile, res=c, N=N)
