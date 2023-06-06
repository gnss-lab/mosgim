import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from mosgim.loader.tec_prepare import MagneticCoordType
from mosgim.plotter.plotN import plot_and_save
from mosgim.mosg.mosgim import calculate_maps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve raw TECs to ')
    parser.add_argument('--in_file', 
                        type=Path, 
                        default=Path('/tmp/lcp.npz'),
                        help='Path to data, after prepare script')
    parser.add_argument('--out_file', 
                        type=Path, 
                        default=Path('/tmp/map.npz'),
                        help='Path to data, after prepare script')
    parser.add_argument('--animation_file', 
                        type=Path, 
                        default=Path('/tmp/animation.mp4'),
                        help='Path to data, after prepare script')
    args = parser.parse_args()
    inputfile = args.in_file
    outputfile = args.out_file
    animation_file = args.animation_file
    data = np.load(inputfile, allow_pickle=True)
    maps = calculate_maps(data['res'], MagneticCoordType.mdip, datetime(2017, 1, 2))
    plot_and_save(maps, animation_file, outputfile)