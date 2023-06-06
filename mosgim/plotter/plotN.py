from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np


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

