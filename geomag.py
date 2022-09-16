import pyIGRF.calculate as calculate
import numpy as np
from geo import subsol
from time_util import sec_of_day
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


@np.vectorize
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

@np.vectorize
def geo2modip(theta, phi, date):
    year = date.year
    I = inclination(lat=np.rad2deg(np.pi/2 - theta), lon=np.rad2deg(phi), alt=300., year=year) # alt=300 for modip300
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
