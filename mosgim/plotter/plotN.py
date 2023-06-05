from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import scipy.special as sp
from mosgim.geo.geomag import geo2mag
from mosgim.geo.geomag import geo2modip
from mosgim.loader.tec_prepare import MagneticCoordType

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


def calculate_maps(res, mag_type, date, **kwargs):
    nbig = kwargs.get('Y_order', 15) 
    mbig = kwargs.get('Y_degree', 15)
    nT = kwargs.get('number_time_steps', 24)
    lat_step = kwargs.get('lat_step', 2.5)
    lon_step = kwargs.get('lat_step', 5.)
    
    # prepare net to estimate TEC on it
    colat = np.arange(2.5, 180, lat_step)
    lon = np.arange(-180, 185, lon_step)
    lon_grid, colat_grid = np.meshgrid(lon, colat)

    maps = {}
    maps['lons'] = lon_grid
    maps['lats'] = 90.-colat_grid
    for k in np.arange(0,nT,1): # consecutive tec map number
        map_time = date + timedelta(0, np.int(k / nT * 86400.) )
        if mag_type == MagneticCoordType.mdip:
            mcolat, mt = geo2modip(np.deg2rad(colat_grid.flatten()), 
                                   np.deg2rad(lon_grid.flatten()), 
                                   map_time)
        elif mag_type == MagneticCoordType.mag:
            mcolat, mt = geo2mag(np.deg2rad(colat_grid.flatten()), 
                                np.deg2rad(lon_grid.flatten()), 
                                map_time)
        else:
            raise ValueError('Unknow magnetic coord type')
        Atest = make_matrix(nbig, mbig, mt, mcolat)
        map_cells = len(Atest[0])
        time_slice = res[k*map_cells: (k+1)*map_cells]
        Z1 = np.dot(Atest, time_slice).reshape(len(colat), len(lon))
        maps['time' + str(k).zfill(2)] = Z1
    return maps
    
def plot_and_save(maps, animation_file, maps_file, **kwargs):
    max_tec = kwargs.get('max_tec', 40)
    maps_keys = [k for k in maps if k[:4] == 'time']
    maps_keys.sort()
    fig = plt.figure()
    camera = Camera(fig)
    levels=np.arange(0, max_tec, 0.5)
    for k in maps_keys:
        plt.contourf(maps['lons'], maps['lats'], maps[k], 
                     levels, cmap=plt.cm.jet)
        camera.snap()
    anim = camera.animate()
    anim.save(animation_file)
    np.savez(maps_file, maps)


if __name__ == '__main__':
    data = np.load('res_data_rel_modip300_2017_002_lcp.npz', allow_pickle=True)
    maps = calculate_maps(data['res'], MagneticCoordType.mdip, datetime(2017, 1, 2))
    plot_and_save(maps, 'animation.mp4', 'maps.npz')
