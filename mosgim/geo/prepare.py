import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter
import os
import itertools
from scipy.integrate import quad
import pyIGRF.calculate as calculate



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
geo2modip = np.vectorize(geo2modip)

# END OF GEOMAGNETIC AND MODIP COORDINATES SECTION



def sec_of_day(time):
    return (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
sec_of_day = np.vectorize(sec_of_day)


def sec_of_interval(time, time0):
    return (time - time0).total_seconds()
sec_of_interval = np.vectorize(sec_of_interval, excluded='time0')


def getContInt(time, tec, lon, lat, el,  maxgap=30, maxjump=1):
    r = np.array(range(len(time)))
    idx = np.isfinite(tec) & np.isfinite(lon) & np.isfinite(lat) & np.isfinite(el) & (el > 10.)
    r = r[idx]
    intervals = []
    if len(r) == 0:
        return intervals
    beginning = r[0]
    last = r[0]
    last_time = time[last]
    for i in r[1:]:
        if abs(time[i] - last_time) > maxgap or abs(tec[i] - tec[last]) > maxjump:
            intervals.append((beginning, last))
            beginning = i
        last = i
        last_time = time[last]
        if i == r[-1]:
            intervals.append((beginning, last))
    return idx, intervals



#################### MAIN SCRIPT GOES HERE#######################################


header = ['datetime', 'el', 'ipp_lat', 'ipp_lon', 'tec']
dtype=zip(header,(object, float, float, float, float))
convert = lambda x: datetime.strptime(x.decode("utf-8"), "%Y-%m-%dT%H:%M:%S")
rootdir = '~/mosgim/002/'
time0 = datetime(2017, 1, 2)
derivative = False



Atec = Atime = Along = Alat = Ael = Atime_ref = Along_ref = Alat_ref = Ael_ref = np.array([])


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".dat"):
            print (filepath)

            try:            
                data = np.genfromtxt(filepath, comments='#', names=header, dtype=(object, float, float, float, float), converters={"datetime": convert},  unpack=True)
                data = {k: arr for k, arr in zip(header, data)}
                tt = sec_of_day(data['datetime'])                
                idx, intervals = getContInt(tt, data['tec'], data['ipp_lon'], data['ipp_lat'], data['el'],  maxgap=35., maxjump=2.)

                print(intervals)

                for ii in intervals:
                    if (tt[ii[1]] - tt[ii[0]]) >= 1 * 60 * 60:    # disgard all the arcs shorter than 1 hour
                        tec_out = savgol_filter(data['tec'][ii[0]:ii[1]], 21, 2) # parabolic smoothing with 10 min window
                        time_out = data['datetime'][ii[0]:ii[1]]
                        ipp_lon_out = data['ipp_lon'][ii[0]:ii[1]]
                        ipp_lat_out = data['ipp_lat'][ii[0]:ii[1]]
                        el_out = data['el'][ii[0]:ii[1]]

                        ind_sparse = (tt[ii[0]:ii[1]] % 600 == 0)
                        tec_out = tec_out[ind_sparse]
                        time_out = time_out[ind_sparse]
                        ipp_lon_out = ipp_lon_out[ind_sparse]
                        ipp_lat_out = ipp_lat_out[ind_sparse]
                        el_out = el_out[ind_sparse]

                        if derivative == True:
                            dtec = tec_out[1:] - tec_out[0:-1]
                            time_out_ref = time_out[0:-1]
                            time_out = time_out[1:]
                            ipp_lon_out_ref = ipp_lon_out[0:-1]
                            ipp_lon_out = ipp_lon_out[1:]
                            ipp_lat_out_ref = ipp_lat_out[0:-1]
                            ipp_lat_out = ipp_lat_out[1:]
                            el_out_ref = el_out[0:-1]
                            el_out = el_out[1:]
                       
                        if derivative == False:

                            idx_min = np.argmin(tec_out)
                            tec0 = tec_out[idx_min]
                            t0 = time_out[idx_min]                            
                            ipp_lon0 = ipp_lon_out[idx_min]
                            ipp_lat0 = ipp_lat_out[idx_min]
                            el0 = el_out[idx_min]                            


                            tec_out = np.delete(tec_out, idx_min)
                            time_out = np.delete(time_out, idx_min)
                            ipp_lon_out = np.delete(ipp_lon_out, idx_min)
                            ipp_lat_out = np.delete(ipp_lat_out, idx_min)
                            el_out = np.delete(el_out, idx_min)


                            dtec = tec_out - tec0
                            time_out_ref = np.array([t0 for _ in range(len(time_out))])
                            ipp_lon_out_ref = ipp_lon0 * np.ones(len(ipp_lon_out))
                            ipp_lat_out_ref = ipp_lat0 * np.ones(len(ipp_lat_out))
                            el_out_ref = el0 * np.ones(len(el_out))

             

                        Atec = np.append(Atec, dtec)
                        Atime = np.append(Atime, time_out)
                        Along = np.append(Along, ipp_lon_out)
                        Alat = np.append(Alat, ipp_lat_out)
                        Ael = np.append(Ael, el_out)
                        Atime_ref = np.append(Atime_ref, time_out_ref)
                        Along_ref = np.append(Along_ref, ipp_lon_out_ref)
                        Alat_ref = np.append(Alat_ref, ipp_lat_out_ref)
                        Ael_ref = np.append(Ael_ref, el_out_ref)



                      
                    else: 
                        print('too short interval')

            except Exception:
                print('warning')


print ('number of observations', len(Atec))



print ('preparing coordinate system')


mcolat, mlt = geo2modip(np.pi/2 - np.deg2rad(Alat), np.deg2rad(Along), Atime)  # modip coordinates in rad
mcolat_ref, mlt_ref = geo2modip(np.pi/2 - np.deg2rad(Alat_ref), np.deg2rad(Along_ref), Atime_ref)  

mcolat1, mlt1 = geo2mag(np.pi/2 - np.deg2rad(Alat), np.deg2rad(Along), Atime)  # geomag coordinates in rad
mcolat1_ref, mlt1_ref = geo2mag(np.pi/2 - np.deg2rad(Alat_ref), np.deg2rad(Along_ref), Atime_ref)  


print ('saving input data')


np.savez('input_data_rel_modip300_2017_002.npz', day=time0,
         time=sec_of_interval(Atime, time0), mlt=mlt, mcolat=mcolat, el=np.deg2rad(Ael),
         time_ref=sec_of_interval(Atime_ref, time0), mlt_ref=mlt_ref, mcolat_ref=mcolat_ref, el_ref=np.deg2rad(Ael_ref), rhs=Atec)



np.savez('input_data_rel_gm_2017_002.npz', day=time0,
         time=sec_of_interval(Atime, time0), mlt=mlt1, mcolat=mcolat1, el=np.deg2rad(Ael),
         time_ref=sec_of_interval(Atime_ref, time0), mlt_ref=mlt1_ref, mcolat_ref=mcolat1_ref, el_ref=np.deg2rad(Ael_ref), rhs=Atec)





