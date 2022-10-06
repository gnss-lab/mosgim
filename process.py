import argparse
import time
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta

from tec_prepare import (DataSourceType,
                         MagneticCoordType,
                         ProcessingType,
                         process_data,
                         combine_data,
                         get_data,
                         save_data,
                         sites,
                         calculate_seed_mag_coordinates_parallel)
from loader import (LoaderHDF, 
                    LoaderTxt)
from mosgim import solve_weights
from createLCP import create_lcp
from plotN import plot_and_save

def __populate_out_path(args):
    date = args.date
    out_path = args.out_path
    if out_path:
        if not args.modip_file:
            args.modip_file = out_path / f'prepared_modip_{date}.npz'
        if not args.mag_file:
            args.mag_file = out_path /f'prepared_mag_{date}.npz'
        if not args.weight_file:
            args.weight_file = out_path / f'weights_{date}.npz'
        if not args.lcp_file:
            args.lcp_file = out_path / f'lcp_{date}.npz'
        if not args.maps_file:
            args.maps_file = out_path / f'maps_{date}.npz'
        if not args.animation_file:
            args.animation_file = out_path / f'animation_{date}.mp4'

def __parser_args(command=''):
    parser = argparse.ArgumentParser(description='Prepare data from txt, hdf, or RInEx')
    parser.add_argument('--data_path', 
                        type=Path, 
                        required=True,
                        help='Path to data, content depends on format')
    parser.add_argument('--out_path', 
                        type=Path, 
                        default=Path('/tmp/'),
                        help='Path where results are stored')
    parser.add_argument('--process_type', 
                        type=ProcessingType, 
                        required=True,
                        help='Type of processing [single | ranged]')
    parser.add_argument('--data_source', 
                        type=DataSourceType, 
                        required=True,
                        help='Path to data, content depends on format, [hdf | txt | rinex]')
    parser.add_argument('--date',  
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        required=True,
                        help='Date of data, example 2017-01-02')
    parser.add_argument('--ndays', 
                        type=int, 
                        required=True,
                        help='Number of days to process in "ranged" processing')
    parser.add_argument('--modip_file',  
                        type=Path,
                        help='Path to file with results, for modip')
    parser.add_argument('--mag_file',  
                        type=Path,
                        help='Path to file with results, for magnetic lat')
    parser.add_argument('--weight_file',  
                        type=Path,
                        help='Path to file with solved weights')
    parser.add_argument('--lcp_file',  
                        type=Path,
                        help='LCP file')
    parser.add_argument('--mag_type',  
                        type=MagneticCoordType,
                        required=True,
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
                        action='store_true',
                        help='Skip data reading use existing files')
    parser.add_argument('--animation_file',  
                        type=Path,
                        help='Path to animation')
    parser.add_argument('--maps_file',  
                        type=Path,
                        help='Path to map data')
    parser.add_argument('--const',  
                        action='store_true',
                        help='Defines ')
    if command:
        args = parser.parse_args(command.split())
    else:
        args = parser.parse_args()
    if args.process_type == ProcessingType.ranged and (args.ndays is None):
        parser.error("Ranged processing requires --ndays")
    if args.process_type == ProcessingType.ranged:
        basetime = args.date
        for iday in range(args.ndays):
            _args = argparse.Namespace(**vars(args))
            d = basetime + timedelta(iday)
            doy = str(d.timetuple().tm_yday).zfill(3)
            _args.date = d
            _args.data_path = f"{args.data_path}/{d.year}/{doy}"
            __populate_out_path(_args)
            yield _args
    elif args.process_type == ProcessingType.single:
        __populate_out_path(args)
        for _args in [args]:
            yield _args
    else:
        pass

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

        elif args.data_source == DataSourceType.txt:
            loader = LoaderTxt(args.data_path)
            data_generator = loader.generate_data_pool(sites=_sites, 
                                                       nworkers=args.nworkers)
        else:
            raise ValueError('Define data source')
        data = process_data(data_generator)
        print(loader.not_found_sites)
        print(f'Done reading in {time.time() - st}')
        data_chunks = combine_data(data, nchunks=args.nworkers)
        print('Start magnetic calculations...')
        st = time.time()
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
                               gigs=args.memory_per_worker,
                               linear= not args.const)
    if args.weight_file:
        np.savez(args.weight_file, res=weights, N=N)
    try:
        lcp = create_lcp({'res': weights, 'N': N})
    except Exception:
        print('Could not finish calculation, LCP is failed')
        return
    if args.lcp_file:
        np.savez(args.lcp_file, res=lcp, N=N)
    plot_and_save(lcp, args.animation_file, args.maps_file)


if __name__ == '__main__':
    for args in __parser_args():
        __process(args)
    
