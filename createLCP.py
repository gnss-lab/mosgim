import sys
from loguru import logger

import numpy as np
import lemkelcp as lcp
import scipy.special as sp
from scipy.sparse import csr_matrix

from tqdm import tqdm


def logger_configuration() -> None:
    logger.remove()

    logger.add(
        sys.stdout, colorize=True, format="(<level>{level}</level>) [<green>{time:HH:mm:ss}</green>] ➤ <level>{message}</level>")


class CreateLCP:
    def __init__(self, nbig: int, mbig: int, nT: int) -> None:
        """
        Parameters
        ----------
        nbig : int
            Max order of spherical harmonic expansion
        mbig : int
            Max degree of spherical harmonic expansion (0 <= mbig <= nbig)
        nT : int
            Number of time steps
        """
        self.__nbig = nbig
        self.__mbig = mbig
        self.__nT = nT

        self.__vcoefs = np.vectorize(self.__calc_coefs, excluded=[
                                     'M', 'N'], otypes=[np.ndarray])

    def __calc_coefs(self, M, N, theta, phi):
        """
        Parameters
        ----------
        M
            Meshgrid of harmonics degrees
        N
            Meshgrid of harmonics orders
        theta
            Array of LTs of IPPs in rads
        phi
            Array of co latitudes of IPPs in rads
        """
        n_coefs = len(M)
        a = np.zeros(n_coefs)

        # complex harmonics on meshgrid
        Ymn = sp.sph_harm(np.abs(M), N, theta, phi)

        #  introducing real basis according to scipy normalization
        a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
        a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
        a[M == 0] = Ymn[M == 0].real

        return a

    def construct(self, theta, phi, timeindex):
        """
        Parameters
        ----------
        theta
            Array of LTs of IPPs in rads
        phi
            Array of co latitudes of IPPs in rads
        timeindex
            Number of time steps
        """

        # Construct matrix of the problem (A)
        n_ind = np.arange(0, self.__nbig + 1, 1)
        m_ind = np.arange(-self.__mbig, self.__mbig + 1, 1)
        M, N = np.meshgrid(m_ind, n_ind)
        Y = sp.sph_harm(np.abs(M), N, 0, 0)
        idx = np.isfinite(Y)
        M = M[idx]
        N = N[idx]
        n_coefs = len(M)

        len_rhs = len(phi)

        a = self.__vcoefs(M=M, N=N, theta=theta, phi=phi)

        logger.info(f"coefs done {n_coefs}")

        # prepare (A) in csr sparse format
        data = np.empty(len_rhs * n_coefs)
        rowi = np.empty(len_rhs * n_coefs)
        coli = np.empty(len_rhs * n_coefs)

        for i in tqdm(range(0, len_rhs, 1)):
            data[i * n_coefs: (i + 1) * n_coefs] = a[i]
            rowi[i * n_coefs: (i + 1) * n_coefs] = i * \
                np.ones(n_coefs).astype('int32')

            coli[i * n_coefs: (i + 1) * n_coefs] = np.arange(timeindex[i]
                                                             * n_coefs, (timeindex[i] + 1) * n_coefs, 1).astype('int32')

        A = csr_matrix((data, (rowi, coli)), shape=(
            len_rhs, (self.__nT + 1) * n_coefs))

        logger.success("matrix (A) done")

        return A


if __name__ == "__main__":
    logger_configuration()

    input_file = 'res_data_rel_modip300_2017_002.npz'
    output_file = 'res_data_rel_modip300_2017_002_lcp.npz'

    nT = 24

    colat = np.arange(2.5, 180, 2.5)
    mlt = np.arange(0., 365., 5.)
    mlt_m, colat_m = np.meshgrid(mlt, colat)

    mlt_m = np.tile(mlt_m.flatten(), nT + 1)
    colat_m = np.tile(colat_m.flatten(), nT + 1)
    time_m = np.array([int(_ / (len(colat) * len(mlt)))
                      for _ in range(len(colat) * len(mlt) * (nT + 1))])

    G = CreateLCP(nbig=15, mbig=15, nT=nT).construct(
        theta=np.deg2rad(mlt_m),
        phi=np.deg2rad(colat_m),
        timeindex=time_m
    )

    logger.info(f"Load data {input_file}")
    data = np.load(input_file, allow_pickle=True)

    Ninv = np.linalg.inv(data['N'])
    w = G.dot(data['res'])
    idx = (w < 0)

    Gnew = G[idx, :]
    wnew = Gnew.dot(data['res'])

    logger.info("constructing M")

    NGT = Ninv * Gnew.transpose()
    M = Gnew.dot(NGT)

    sol = lcp.lemkelcp(M, wnew, 10000)

    c = data['res'] + NGT.dot(sol[0])
    w = G.dot(c)

    np.savez(output_file, res=c, N=data['N'])

    logger.success(f"{output_file} saved successfully")
