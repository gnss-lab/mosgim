from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.special as sp
import re
import pyIGRF.calculate as calculate


def sec_of_day(time):
    return (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
sec_of_day = np.vectorize(sec_of_day)


# GEOMAGNETIC AND MODIP COORDINATES SECTION

# North magnetic pole coordinates, for 2017
# Taken from here: http://wdc.kugi.kyoto-u.ac.jp/poles/polesexp.html
POLE_THETA = np.pi/2 - np.radians(80.5)
POLE_PHI = np.radians(-72.6)

# Geodetic to Geomagnetic transform: http://www.nerc-bas.ac.uk/uasd/instrums/magnet/gmrot.html
GEOGRAPHIC_TRANSFORM = np.array([
    [np.cos(POLE_THETA)*np.cos(POLE_PHI), np.cos(POLE_THETA)*np.sin(POLE_PHI), -np.sin(POLE_THETA)],
    [-np.sin(POLE_PHI), np.cos(POLE_PHI), 0],
    [np.sin(POLE_THETA)*np.cos(POLE_PHI), np.sin(POLE_THETA)*np.sin(POLE_PHI), np.cos(POLE_THETA)]
])


def subsol(year, doy, ut):
    '''Finds subsolar geocentric longitude and latitude.


    Parameters
    ==========
    year : int [1601, 2100]
        Calendar year
    doy : int [1, 365/366]
        Day of year
    ut : float
        Seconds since midnight on the specified day

    Returns
    =======
    sbsllon : float
        Subsolar longitude [rad] for the given date/time
    sbsllat : float
        Subsolar co latitude [rad] for the given date/time

    Notes
    =====

    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994). Usable for years 1601-2100,
    inclusive. According to the Almanac, results are good to at least 0.01
    degree latitude and 0.025 degrees longitude between years 1950 and 2050.
    Accuracy for other years has not been tested. Every day is assumed to have
    exactly 86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored (their effect is below the accuracy threshold of the
    algorithm).

    After Fortran code by A. D. Richmond, NCAR. Translated from IDL
    by K. Laundal.

    '''

    from numpy import sin, cos, pi, arctan2, arcsin

    yr = year - 2000

    if year >= 2101:
        print('subsol.py: subsol invalid after 2100. Input year is:', year)

    nleap = np.floor((year-1601)/4)
    nleap = nleap - 99
    if year <= 1900:
        if year <= 1600:
            print('subsol.py: subsol invalid before 1601. Input year is:', year)
        ncent = np.floor((year-1601)/100)
        ncent = 3 - ncent
        nleap = nleap + ncent

    l0 = -79.549 + (-0.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-0.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400 - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = 0.9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = 0.9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*pi/180

    # Ecliptic longitude:
    lmbda = l + 1.915*sin(grad) + 0.020*sin(2*grad)
    lmrad = lmbda*pi/180
    sinlm = sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4e-7*n
    epsrad = epsilon*pi/180

    # Right ascension:
    alpha = arctan2(cos(epsrad)*sinlm, cos(lmrad)) * 180/pi

    # Declination:
    delta = arcsin(sin(epsrad)*sinlm) * 180/pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = round(etdeg/360)
    etdeg = etdeg - 360*nrot

    # Apparent time (degrees):
    aptime = ut/240 + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180 - aptime
    nrot = round(sbsllon/360)
    sbsllon = sbsllon - 360*nrot

    return np.deg2rad(sbsllon), np.pi / 2 - np.deg2rad(sbsllat)


def geo2mag(theta, phi, date):

    ut = sec_of_day(date)
    doy = date.timetuple().tm_yday
    year = date.year

    phi_sbs, theta_sbs = subsol(year, doy, ut)
    r_sbs = np.array([np.sin(theta_sbs) * np.cos(phi_sbs), np.sin(theta_sbs) * np.sin(phi_sbs), np.cos(theta_sbs)])

    r_sbs_mag = GEOGRAPHIC_TRANSFORM.dot(r_sbs)
    theta_sbs_m = np.arccos(r_sbs_mag[2])
    phi_sbs_m = np.arctan2(r_sbs_mag[1], r_sbs_mag[0])
    if phi_sbs_m < 0.:
        phi_sbs_m = phi_sbs_m + 2. * np.pi


    r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    r_mag = GEOGRAPHIC_TRANSFORM.dot(r)
    theta_m = np.arccos(r_mag[2])
    phi_m = np.arctan2(r_mag[1], r_mag[0])
    if phi_m < 0.:
        phi_m = phi_m + 2. * np.pi


    mlt = phi_m - phi_sbs_m + np.pi # np.radians(15.) * ut /3600. + phi_m + POLE_PHI  
    if mlt < 0.:
        mlt = mlt + 2. * np.pi
    if mlt > 2. * np.pi:
        mlt = mlt - 2. * np.pi

    return theta_m, mlt
geo2mag = np.vectorize(geo2mag)


def inclination(lat, lon, alt=300., year=2005.):
    """
    :return
         I is inclination (+ve down)
    """
    if lon < 0:
        lon = lon + 360.      
    FACT = 180./np.pi
    REm = 6371.2
    x, y, z, f = calculate.igrf12syn(year, 2, REm + alt, lat, lon) # 2 stands for geocentric coordinates
    h = np.sqrt(x * x + y * y)
    i = FACT * np.arctan2(z, h)
    return i


def geo2modip(theta, phi, date):
    year = date.year
    I = inclination(lat=np.rad2deg(np.pi/2 - theta), lon=np.rad2deg(phi), alt=300., year=year)
    theta_m = np.pi/2 - np.arctan2(np.deg2rad(I), np.sqrt(np.cos(np.pi/2 - theta)))
    ut = sec_of_day(date)
    phi_sbs = np.deg2rad(180. - ut*15./3600)
    if phi_sbs < 0.:
        phi_sbs = phi_sbs + 2. * np.pi
    if phi < 0.:
        phi = phi + 2. * np.pi
    mlt = phi - phi_sbs + np.pi
    if mlt < 0.:
        mlt = mlt + 2. * np.pi
    if mlt > 2. * np.pi:
        mlt = mlt - 2. * np.pi
    return theta_m, mlt
geo2modip = np.vectorize(geo2modip)




def make_matrix(nbig, mbig, theta, phi):
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sp.sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
    matrix = np.zeros((len(theta), n_coefs))
    for i in range(0, len(theta), 1):
        Ymn = sp.sph_harm(np.abs(M), N, theta[i], phi[i])
        a = np.zeros(len(Ymn))
        a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
        a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
        a[M == 0] = Ymn[M == 0].real
        matrix[i, :] = a[:]
    return matrix
 
def plot_and_save(res, animation_file, maps_file):

    nbig = 15  # max order of spherical harmonic expansion
    mbig = 15  # max degree of spherical harmonic expansion (0 <= mbig <= nbig)
    nT = 24  # number of time steps
    
    # prepare net to estimate TEC on it
    colat = np.arange(2.5, 180, 2.5)
    lon = np.arange(-180, 185, 5.)
    lon_m, colat_m = np.meshgrid(lon, colat)
    fig = plt.figure()
    camera = Camera(fig)
    levels=np.arange(0,40,0.5)
    maps = {}
    maps['lons'] = lon_m
    maps['lats'] = colat_m
    for k in np.arange(0,nT,1): # consecutive tec map number
    #  mcolat, mt = geo2modip(np.deg2rad(colat_m.flatten()), np.deg2rad(lon_m.flatten()), datetime.datetime(2017,1,2, np.int(24 * k / nT),0))
        mcolat, mt = geo2mag(np.deg2rad(colat_m.flatten()), np.deg2rad(lon_m.flatten()), datetime.datetime(2017,1,2, np.int(24 * k / nT),0))
        Atest = make_matrix(nbig, mbig, mt, mcolat)
        Z1 = np.dot(Atest, res[(0+k)*len(Atest[0]):(0+k+1)*len(Atest[0])]).reshape(len(colat), len(lon))
        plt.contourf(lon_m, 90.-colat_m, Z1, levels, cmap=plt.cm.jet)
        maps['time' + k] = Z1
        camera.snap()

    anim = camera.animate()
    anim.save(animation_file)
    np.savez(maps_file, maps)

if __name__ == '__main__':
    data = np.load('res_data_rel_modip300_2017_002_lcp.npz', allow_pickle=True)
    plot_and_save(data['res'], 'animation.mp4', 'maps.npz')
