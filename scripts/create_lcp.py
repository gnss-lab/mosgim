import argparse
import numpy as np

from pathlib import Path
from loguru import logger

from mosgim.mosg.lcp_solver import create_lcp as crelcp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve raw TECs to ')
    parser.add_argument('--in_file', 
                        type=Path, 
                        default=Path('/tmp/mosgim_weights.npz'),
                        help='Path to data, after prepare script')
    parser.add_argument('--out_file', 
                        type=Path, 
                        default=Path('/tmp/lcp.npz'),
                        help='Path to data, after prepare script')
    args = parser.parse_args()
    inputfile = args.in_file
    outputfile = args.out_file
    data = np.load(inputfile, allow_pickle=True)
    c = crelcp(data)
    np.savez(outputfile, res=c, N=data['N'])

    logger.success(f"{outputfile} saved successfully")