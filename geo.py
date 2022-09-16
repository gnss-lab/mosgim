from numpy import sin, cos, pi, arctan2, arcsin, floor, deg2rad

RE = 6371.2
HM=300

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

    yr = year - 2000

    if year >= 2101:
        print('subsol.py: subsol invalid after 2100. Input year is:', year)

    nleap = floor((year-1601)/4)
    nleap = nleap - 99
    if year <= 1900:
        if year <= 1600:
            print('subsol.py: subsol invalid before 1601. Input year is:', year)
        ncent = floor((year-1601)/100)
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

    return deg2rad(sbsllon), pi / 2 - deg2rad(sbsllat)


def sub_ionospheric(s_lat, s_lon, hm, az, el, R=RE):
    """
    Calculates subionospheric point and delatas from site
    Parameters:
        s_lat, slon - site latitude and longitude in radians
        hm - ionposheric maximum height (km)
        az, el - azimuth and elevation of the site-sattelite line of sight in
            radians
        R - Earth radius (km)
    """
    #TODO use meters
    psi = pi / 2 - el - arcsin(cos(el) * R / (R + hm))
    lat = bi = arcsin(sin(s_lat) * cos(psi) + cos(s_lat) * sin(psi) * cos(az))
    lon = sli = s_lon + arcsin(sin(psi) * sin(az) / cos(bi))
    
    lon[lon > pi] = lon[lon > pi] - 2 * pi 
    lon[lon < -pi] = lon[lon < -pi] + 2 * pi 
    return lat, lon
