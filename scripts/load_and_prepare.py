import time
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from mosgim.data.loader import (LoaderTxt, 
                                LoaderHDF)
from mosgim.data.tec_prepare import (process_data,
                                     combine_data,
                                     calculate_seed_mag_coordinates_parallel,
                                     save_data,
                                     sites,
                                     DataSourceType)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data from txt, hdf, or RInEx')
    parser.add_argument('--data_path', 
                        type=Path, 
                        help='Path to data, content depends on format')
    parser.add_argument('--data_source', 
                        type=DataSourceType, 
                        help='Path to data, content depends on format')
    parser.add_argument('--date',  
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        help='Date of data, example 2017-01-02')
    parser.add_argument('--modip_file',  
                        type=Path,
                        default=Path('/tmp/prepared_modip.npz'),
                        help='Path to file with results, for modip')
    parser.add_argument('--mag_file',  
                        type=Path,
                        default=Path('/tmp/prepared_mag.npz'),
                        help='Path to file with results, for magnetic lat')
    parser.add_argument('--nsite',  
                        type=int,
                        help='Number of sites to take into calculations')
    args = parser.parse_args()
    process_date = args.date
    if args.nsite:
        sites = sites[:args.nsite]
        
    if args.data_source == DataSourceType.hdf:
        loader = LoaderHDF(args.data_path)
        data_generator = loader.generate_data(sites=sites)
    if args.data_source == DataSourceType.txt:
        loader = LoaderTxt(args.data_path)
        data_generator = loader.generate_data(sites=sites)
    data = process_data(data_generator)
    data_chunks = combine_data(data, nchunks=1)
    print('Start magnetic calculations...')
    st = time.time()
    #calc_mag_coordinates(combined_data)
    result = calculate_seed_mag_coordinates_parallel(data_chunks)
    print(f'Done, took {time.time() - st}')
    save_data(result, args.modip_file, args.mag_file, process_date)