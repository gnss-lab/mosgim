import argparse
import time
from pathlib import Path
from datetime import datetime
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

if __name__ == '__main__':
    
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
                        default=Path('/tmp/prepared_modip.npz'),
                        help='Path to file with results, for modip')
    parser.add_argument('--mag_file',  
                        type=Path,
                        default=Path('/tmp/prepared_mag.npz'),
                        help='Path to file with results, for magnetic lat')
    parser.add_argument('--mag_type',  
                        type=MagneticCoordType,
                        help='Type of magnetic coords [mag | mdip]')
    args = parser.parse_args()
    process_date = args.date
    if args.data_source == DataSourceType.hdf:
        loader = LoaderHDF(args.data_path)
        data_generator = loader.generate_data(sites=sites)
    if args.data_source == DataSourceType.txt:
        loader = LoaderTxt(args.data_path)
        data_generator = loader.generate_data(sites=sites)
    data = process_data(data_generator)
    data_chunks = combine_data(data, nchunks=3)
    print('Start magnetic calculations...')
    st = time.time()
    #calc_mag_coordinates(combined_data)
    result = calculate_seed_mag_coordinates_parallel(data_chunks)
    print(f'Done, took {time.time() - st}')
    if args.mag_file and args.modip_file:
        save_data(result, args.modip_file, args.mag_file, process_date)
    data = get_data(result, args.mag_type, process_date)
    weights, N = solve_weights(data)
    lcp = create_lcp({'res': weights, 'N': N})
    plot_and_save(lcp, '/tmp/animation.mp4')
    
