import argparse
import numpy as np

from pathlib import Path
from mosgim.mosg.map_creator import solve_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve raw TECs to ')
    parser.add_argument('--in_file', 
                        type=Path, 
                        default=Path('/tmp/prepared_modip.npz'),
                        help='Path to data, after prepare script')
    parser.add_argument('--out_file', 
                        type=Path, 
                        default=Path('/tmp/mosgim_weights.npz'),
                        help='Path to data, after prepare script')
    args = parser.parse_args()
    inputfile = args.in_file
    outputfile = args.out_file
    data = np.load(inputfile, allow_pickle=True)
    res, N = solve_weights(data)
    np.savez(outputfile, res=res, N=N)
