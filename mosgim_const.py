import numpy as np
import scipy.special as sp
from scipy.linalg import solve
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import datetime
import gc


RE = 6371200.
IPPh = 450000.
nbig = 15  # max order of spherical harmonic expansion
mbig = 15  # max degree of spherical harmonic expansion (0 <= mbig <= nbig)
nT = 24  # number of time steps
ndays = 1
sigma0 = 0.075  # TECU - measurement noise at zenith
sigma_v = 1.  # TECU - allowed variability for each coef between two consecutive maps
 

inputfile = 'input_nequick_der(gps+glo)_2017_001.npz'
outputfile = 'res_nequick_der(gps+glo)_2017_001.npy'
 
def MF(el):
    """
    :param el: elevation angle in rads
    """
    return 1./np.sqrt(1 - (RE * np.cos(el) / (RE + IPPh)) ** 2)
 

def calc_coefs(M, N, theta, phi, sf):
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
    return a*sf
vcoefs = np.vectorize(calc_coefs, excluded=['M','N'], otypes=[np.ndarray])
 


def construct_normal_system(nbig, mbig, nT, ndays, time, theta, phi, el, time_ref, theta_ref, phi_ref, el_ref, rhs):
    """
    :param nbig: maximum order of spherical harmonic
    :param mbig: maximum degree of spherical harmonic
    :param nT: number of time intervals
    :param ndays: number of days in analysis
    :param time: array of times of IPPs in secs
    :param theta: array of LTs of IPPs in rads
    :param phi: array of co latitudes of IPPs in rads
    :param el: array of elevation angles in rads
    :param time_ref: array of ref times of IPPs in sec
    :param theta_ref: array of ref longitudes (LTs) of IPPs in rads
    :param phi_ref: array of ref co latitudes of IPPs in rads
    :param el_ref: array of ref elevation angles in rads
    :param rhs: array of rhs (measurements TEC difference on current and ref rays)
    """
    print('constructing normal system for series')
    
    SF = MF(el)
    SF_ref = MF(el_ref)
 
    # Construct weight matrix for the observations
    len_rhs = len(rhs)
    P = lil_matrix((len_rhs, len_rhs))
    diagP = (np.sin(np.deg2rad(el))**2) * (np.sin(np.deg2rad(el_ref))**2) / (np.sin(np.deg2rad(el))**2 + np.sin(np.deg2rad(el_ref))**2)
    P.setdiag(diagP)
    P = P.tocsr()
 
    # Construct matrix of the problem (A)
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sp.sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
 
    timeindex = (time * nT / (ndays * 86400.)).astype('int16')
    timeindex_ref = (time_ref * nT / (ndays * 86400.)).astype('int16')
   
 
    a = vcoefs(M=M, N=N, theta=theta, phi=phi, sf=SF)
    a_ref = vcoefs(M=M, N=N, theta=theta_ref, phi=phi_ref, sf=SF_ref)
    print('coefs done', n_coefs, nT, ndays, len_rhs)

    del theta
    del phi
    del el
    del theta_ref
    del phi_ref
    del el_ref
    del SF
    del SF_ref
    del M
    del N
    del Y
    del diagP
    gc.collect()

    #prepare (A) in csr sparse format
    data = np.empty(len_rhs * n_coefs * 2)
    rowi = np.empty(len_rhs * n_coefs * 2)
    coli = np.empty(len_rhs * n_coefs * 2)

    for i in range(0, len_rhs,1): 
        data[i * n_coefs * 2: i * n_coefs * 2 + n_coefs] = a[i]
        data[i * n_coefs * 2 + n_coefs: (i+1) * n_coefs * 2] = -a_ref[i]
        rowi[i * n_coefs * 2: (i+1) * n_coefs * 2] = i * np.ones(n_coefs * 2).astype('int32')
        coli[i * n_coefs * 2: i * n_coefs * 2 + n_coefs] = np.arange(timeindex[i] * n_coefs, (timeindex[i] + 1) * n_coefs, 1).astype('int32')
        coli[i * n_coefs * 2 + n_coefs: (i+1) * n_coefs * 2] = np.arange(timeindex_ref[i] * n_coefs,(timeindex_ref[i] + 1) * n_coefs, 1).astype('int32')
    
    A = csr_matrix((data, (rowi, coli)), shape=(len_rhs, nT * n_coefs))
    print('matrix (A) for subset done')



 
    del a
    del a_ref    
    del data
    del rowi
    del coli
    del time
    del time_ref

    gc.collect()
    
    # define normal system
    AP = A.transpose().dot(P)
    N = AP.dot(A).todense()    
    b = AP.dot(rhs)

    print('normal matrix (N) done')


    del A
    del AP
    del rhs  
    gc.collect()

    return N, b


def stack_weight_solve_ns(nbig, mbig, nT, ndays, time_chunks, mlt_chunks, mcolat_chunks, el_chunks, 
                          time_ref_chunks, mlt_ref_chunks, mcolat_ref_chunks, el_ref_chunks, rhs_chunks):

    n_coefs = (nbig + 1)**2 - (nbig - mbig) * (nbig - mbig + 1)
    N = np.zeros((n_coefs * (nT), n_coefs * (nT)))
    b = np.zeros(n_coefs * (nT))

    for c_time, c_mlt, c_mcolat, c_el, c_time_ref, c_mlt_ref, c_mcolat_ref, c_el_ref, c_rhs  in zip(time_chunks, mlt_chunks, mcolat_chunks, el_chunks, time_ref_chunks, 
                                                                                                               mlt_ref_chunks, mcolat_ref_chunks, el_ref_chunks, rhs_chunks):
    
        NN, bb = construct_normal_system(nbig, mbig, nT, ndays, c_time, c_mlt, c_mcolat, c_el, c_time_ref, c_mlt_ref, c_mcolat_ref, c_el_ref, c_rhs) 

        N += NN
        b += bb

    print('normal matrix (N) stacked')


    # imposing frozen conditions on consequitive maps coeffs
    for ii in range(0, nT-1, 1):
        for kk in range(0, n_coefs):
            N[ii*n_coefs + kk, ii*n_coefs + kk] += (sigma0 / sigma_v)**2
            N[(ii + 1) * n_coefs + kk, (ii+1) * n_coefs + kk] += (sigma0 / sigma_v)**2
            N[(ii + 1) * n_coefs + kk, ii * n_coefs + kk] += -(sigma0 / sigma_v)**2
            N[ii * n_coefs + kk, (ii + 1) * n_coefs + kk] += -(sigma0 / sigma_v)**2
    print('normal matrix (N) constraints added')

    # # solve normal system
    res1 = solve(N, b)  
    print('normal system solved')
    
    return res1, N


if __name__ == '__main__':

    #n_coefs = (nbig + 1)**2 - (nbig - mbig) * (nbig - mbig + 1)

    # load data
    data = np.load(inputfile, allow_pickle=True)

    time = data['time']
    mlt = data['mlt']
    mcolat = data['mcolat']
    el = data['el']
    time_ref = data['time_ref']
    mlt_ref = data['mlt_ref']
    mcolat_ref = data['mcolat_ref']
    el_ref = data['el_ref']
    rhs = data['rhs']

    nchunks = np.int(len(rhs) / 60000) # set chuncks size to fit in memory ~4Gb

    print('start, nbig=%s, mbig=%s, nT=%s, ndays=%s, sigma0=%s, sigma_v=%s, number of observations=%s, number of chuncks=%s' % (nbig, mbig, nT, ndays, sigma0, sigma_v, len(rhs), nchunks))


    # split data into chunks
    time_chunks = np.array_split(time, nchunks)
    mlt_chunks = np.array_split(mlt, nchunks)
    mcolat_chunks = np.array_split(mcolat, nchunks)
    el_chunks = np.array_split(el, nchunks)
    time_ref_chunks = np.array_split(time_ref, nchunks)
    mlt_ref_chunks = np.array_split(mlt_ref, nchunks)
    mcolat_ref_chunks = np.array_split(mcolat_ref, nchunks)
    el_ref_chunks = np.array_split(el_ref, nchunks)
    rhs_chunks = np.array_split(rhs, nchunks)

    res, N = stack_weight_solve_ns(nbig, mbig, nT, ndays, time_chunks, mlt_chunks, mcolat_chunks, el_chunks, 
                                time_ref_chunks, mlt_ref_chunks, mcolat_ref_chunks, el_ref_chunks, rhs_chunks) 


    np.savez(outputfile, res=res, N=N)
