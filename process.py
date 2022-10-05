import argparse
import time
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta

from tec_prepare import (DataSourceType,
                         MagneticCoordType,
                         process_data,
                         combine_data,
                         get_data,
                         save_data,
                         sites,
                         calculate_seed_mag_coordinates_parallel)
from loader import (LoaderHDF, 
                    LoaderTxt)
from mosgim_linear import solve_weights
from createLCP import create_lcp
from plotN import plot_and_save

def __parser_args(command=''):
    parser = argparse.ArgumentParser(description='Prepare data from txt, hdf, or RInEx')
    parser.add_argument('--data_path', 
                        type=Path, 
                        help='Path to data, content depends on format')
    parser.add_argument('--data_source', 
                        type=DataSourceType, 
                        help='Path to data, content depends on format, [hdf | txt | rinex]')
    parser.add_argument('--date',  
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        help='Date of data, example 2017-01-02')
    parser.add_argument('--modip_file',  
                        type=Path,
                        #default=Path('/tmp/prepared_modip.npz'),
                        help='Path to file with results, for modip')
    parser.add_argument('--mag_file',  
                        type=Path,
                        #default=Path('/tmp/prepared_mag.npz'),
                        help='Path to file with results, for magnetic lat')
    parser.add_argument('--weight_file',  
                        type=Path,
                        #default=Path('/tmp/weights.npz'),
                        help='Path to file with solved weights')
    parser.add_argument('--lcp_file',  
                        type=Path,
                        #default=Path('/tmp/lcp.npz'),
                        help='LCP file')
    parser.add_argument('--mag_type',  
                        type=MagneticCoordType,
                        help='Type of magnetic coords [mag | mdip]')
    parser.add_argument('--nsite',  
                        type=int,
                        help='Number of sites to take into calculations')
    parser.add_argument('--nworkers',  
                        type=int,
                        default=1,
                        help='Numbernworkers of threads for parallel procesing')
    parser.add_argument('--memory_per_worker',  
                        type=int,
                        default=2,
                        help='Number of Gb per worker')
    parser.add_argument('--skip_prepare',
                        help='Skip data reading use existing files')
    parser.add_argument('--animation_file',  
                        type=Path,
                        #default=Path('/tmp/animation.mp4'),
                        help='Path to animation')
    parser.add_argument('--maps_file',  
                        type=Path,
                        #default=Path('/tmp/maps.npz'),
                        help='Path to map data')
    if command:
        args = parser.parse_args(command.split())
    else:
        args = parser.parse_args()
    process_date = args.date
    if not args.modip_file:
        args.modip_file = f'/tmp/prepared_modip_{process_date}.npz'
    if not args.mag_file:
        args.mag_file = f'/tmp/prepared_mag_{process_date}.npz'
    if not args.weight_file:
        args.weight_file = f'/tmp/weights_{process_date}.npz'
    if not args.lcp_file:
        args.lcp_file = f'/tmp/lcp_{process_date}.npz'
    if not args.maps_file:
        args.maps_file = f'/tmp/maps_{process_date}.npz'
    if not args.animation_file:
        args.animation_file = f'/tmp/animation_{process_date}.mp4'
    return args

def __process(args):
    print(args)
    process_date = args.date
    st = time.time()
    if not args.skip_prepare:
        if args.nsite:
            _sites = sites[:args.nsite]
        else:
            _sites = sites[:]
        if args.data_source == DataSourceType.hdf:
            loader = LoaderHDF(args.data_path)
            data_generator = loader.generate_data(sites=_sites)

        if args.data_source == DataSourceType.txt:
            loader = LoaderTxt(args.data_path)
            #data_generator = loader.generate_data(sites=_sites)
            data_generator = loader.generate_data_pool(sites=_sites, 
                                                       nworkers=args.nworkers)
        data = process_data(data_generator)
        print(loader.not_found_sites)
        print(f'Done reading in {time.time() - st}')
        data_chunks = combine_data(data, nchunks=args.nworkers)
        print('Start magnetic calculations...')
        st = time.time()
        #calc_mag_coordinates(combined_data)
        result = calculate_seed_mag_coordinates_parallel(data_chunks, 
                                                        nworkers=args.nworkers)
        print(f'Done, took {time.time() - st}')
        
        if args.mag_file and args.modip_file:
            save_data(result, args.modip_file, args.mag_file, process_date)
        data = get_data(result, args.mag_type, process_date)
    else:
        if args.mag_type == MagneticCoordType.mag:
            data = np.load(args.mag_file, allow_pickle=True)
        elif args.mag_type == MagneticCoordType.mdip:
            data = np.load(args.modip_file, allow_pickle=True)
    weights, N = solve_weights(data, 
                               nworkers=args.nworkers, 
                               gigs=args.memory_per_worker)
    if args.weight_file:
        np.savez(args.weight_file, res=weights, N=N)
    lcp = create_lcp({'res': weights, 'N': N})
    if args.lcp_file:
        np.savez(args.lcp_file, res=lcp, N=N)
    plot_and_save(lcp, args.animation_file, args.maps_file)
    
def __process_range(basetime, ndays):
    for iday in range(1, ndays):
        d = basetime + timedelta(iday-1)
        doy = str(d.timetuple().tm_yday).zfill(3)
        str_date =  datetime.strftime(d, '%Y-%m-%d')
        cmd = f"--data_path /media/user/ssd/mosgim/dat/{d.year}/{doy} " 
        cmd += f"--data_source txt --date {str_date} --mag_type mag "
        cmd += f"--memory_per_worker 4 --nworkers 1"
        args = __parser_args(command=cmd)
        __process(args)
        

if __name__ == '__main__':
    #args = __parser_args()
    #__process(args)
    __process_range(datetime(2017, 1, 1), 32)
    
