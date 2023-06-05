import numpy as np
import scipy.special as sp
import concurrent.futures
import itertools
import datetime
import gc

from scipy.linalg import solve
from scipy.sparse import lil_matrix, csr_matrix
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

RE = 6371200.
IPPh = 450000.
nbig = 15  # max order of spherical harmonic expansion
mbig = 15  # max degree of spherical harmonic expansion (0 <= mbig <= nbig)
nT = 24  # number of time steps
ndays = 1
sigma0 = 0.075  # TECU - measurement noise at zenith
sigma_v = 0.015  # TECU - allowed variability for each coef between two consecutive maps
 
GB_CHUNK = 15000 

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
 


def construct_normal_system(nbig, mbig, nT, ndays, 
                            time, theta, phi, el, 
                            time_ref, theta_ref, phi_ref, el_ref, rhs,
                            linear):
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
    :param linear: bool defines const or linear
    """
    print('constructing normal system for series')
    tmc = time
    tmr = time_ref
    SF = MF(el)
    SF_ref = MF(el_ref)
 
    # Construct weight matrix for the observations
    len_rhs = len(rhs)
    P = lil_matrix((len_rhs, len_rhs))
    el_sin = np.sin(el)
    elr_sin = np.sin(el_ref)
    diagP = (el_sin ** 2) * (elr_sin ** 2) / (el_sin ** 2 + elr_sin **2)
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
 
    tic = (tmc * nT / (ndays * 86400.)).astype('int16')
    tir = (tmr * nT / (ndays * 86400.)).astype('int16')

    ac = vcoefs(M=M, N=N, theta=theta, phi=phi, sf=SF)
    ar = vcoefs(M=M, N=N, theta=theta_ref, phi=phi_ref, sf=SF_ref)
    print('coefs done', n_coefs, nT, ndays, len_rhs)

    #prepare (A) in csr sparse format
    dims = 4 if linear else 2
    nT_add = 1 if linear else 0
    data = np.empty(len_rhs * n_coefs * dims)
    rowi = np.empty(len_rhs * n_coefs * dims)
    coli = np.empty(len_rhs * n_coefs * dims)
    
    if linear:
        hour_cc = (ndays * 86400.) * tic / nT    
        hour_cn = (ndays * 86400.) * (tic + 1) / nT    
        hour_rc = (ndays * 86400.) * tir / nT    
        hour_rn = (ndays * 86400.) * (tir + 1) / nT  
        dt = (ndays * 86400.) / nT 
        for i in range(0, len_rhs, 1): 
            st  = [i * n_coefs * 4 + j * n_coefs for j in range(0, 4)]
            end = [i * n_coefs * 4 + j * n_coefs for j in range(1, 5)]
            data[st[0]: end[0]] =   (  tmc[i] - hour_cc[i]) * ac[i] / dt
            data[st[1]: end[1]] =   ( -tmc[i] + hour_cn[i]) * ac[i] / dt
            data[st[2]: end[2]] = - (  tmr[i] - hour_rc[i]) * ar[i] / dt
            data[st[3]: end[3]] = - ( -tmr[i] + hour_rn[i]) * ar[i] / dt

            rowi[st[0]: end[-1]] = i * np.ones(n_coefs * 4).astype('int32')
            
            coli[st[0]: end[0]] = \
                np.arange((tic[i] + 0) * n_coefs, (tic[i] + 1) * n_coefs, 1).astype('int32')
            coli[st[1]: end[1]] = \
                np.arange((tic[i] + 1) * n_coefs, (tic[i] + 2) * n_coefs, 1).astype('int32')
            coli[st[2]: end[2]] = \
                np.arange((tir[i] + 0) * n_coefs, (tir[i] + 1) * n_coefs, 1).astype('int32')
            coli[st[3]: end[3]] = \
                np.arange((tir[i] + 1) * n_coefs, (tir[i] + 2) * n_coefs, 1).astype('int32')
    else:
        for i in range(0, len_rhs, 1): 
            st  = [i * n_coefs * 2 + j * n_coefs for j in range(0, 2)]
            end = [i * n_coefs * 2 + j * n_coefs for j in range(1, 3)]
            data[st[0]: end[0]] =  ac[i]
            data[st[1]: end[1]] = -ar[i]
            
            rowi[st[0]: end[-1]] = i * np.ones(n_coefs * 2).astype('int32')
            
            coli[st[0]: end[0]] = \
                np.arange(tic[i] * n_coefs, (tic[i] + 1) * n_coefs, 1).astype('int32')
            coli[st[1]: end[1]] = \
                np.arange(tir[i] * n_coefs, (tir[i] + 1) * n_coefs, 1).astype('int32')

    A = csr_matrix((data, (rowi, coli)), 
                   shape=(len_rhs, (nT + nT_add) * n_coefs))
    print('matrix (A) for subset done')
 
    # define normal system
    AP = A.transpose().dot(P)
    N = AP.dot(A).todense()    
    b = AP.dot(rhs)
    print('normal matrix (N) for subset done')

    return N, b


def stack_weight_solve_ns(nbig, mbig, nT, ndays, 
                          time_chunks, mlt_chunks, mcolat_chunks, el_chunks, 
                          time_ref_chunks, mlt_ref_chunks, mcolat_ref_chunks, el_ref_chunks, 
                          rhs_chunks,
                          nworkers=3, 
                          linear=True):

    nT_add = 1 if linear else 0
    n_coefs = (nbig + 1)**2 - (nbig - mbig) * (nbig - mbig + 1)
    N = np.zeros((n_coefs * (nT + nT_add), n_coefs * (nT + nT_add)))
    b = np.zeros(n_coefs * (nT + nT_add))
    
    chunks_processed = []
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        queue = []
        for chunk  in zip(time_chunks, mlt_chunks, mcolat_chunks, el_chunks, 
                          time_ref_chunks, mlt_ref_chunks, mcolat_ref_chunks,
                          el_ref_chunks, rhs_chunks):
            params = (nbig, mbig, nT, ndays) + chunk + (linear, )
            query = executor.submit(construct_normal_system, *params)
            queue.append(query)
        for v in concurrent.futures.as_completed(queue):
            NN, bb = v.result()
            N += NN
            b += bb

    print('normal matrix (N) stacked')


    # imposing frozen conditions on consequitive maps coeffs
    for ii in range(0, nT - 1 + nT_add, 1):
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

def solve_weights(data, gigs=2, nworkers=3, linear=True):
    chunk_size = GB_CHUNK * gigs
    time = data['time']
    mlt = data['mlt']
    mcolat = data['mcolat']
    el = data['el']
    time_ref = data['time_ref']
    mlt_ref = data['mlt_ref']
    mcolat_ref = data['mcolat_ref']
    el_ref = data['el_ref']
    rhs = data['rhs']

    nchunks = np.int(len(rhs) / chunk_size) # set chuncks size to fit in memory ~4Gb
    nchunks = 1 if nchunks < 1 else nchunks

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

    res, N = stack_weight_solve_ns(nbig, mbig, nT, ndays, time_chunks, 
                                   mlt_chunks, mcolat_chunks, el_chunks, 
                                   time_ref_chunks, mlt_ref_chunks, 
                                   mcolat_ref_chunks, el_ref_chunks, rhs_chunks,
                                   nworkers=nworkers,
                                   linear=linear) 
    return res, N


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Solve raw TECs to ')
    parser.add_argument('--in_file', 
                        type=Path, 
                        help='Path to data, after prepare script')
    parser.add_argument('--out_file', 
                        type=Path, 
                        help='Path to data, after prepare script')
    args = parser.parse_args()
    inputfile = args.in_file
    outputfile = args.out_file
    data = np.load(inputfile, allow_pickle=True)
    res, N = solve_weights(data)
    np.savez(outputfile, res=res, N=N)
