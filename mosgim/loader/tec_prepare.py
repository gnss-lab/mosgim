import numpy as np
import time
import concurrent.futures

from numpy import deg2rad as rad
from datetime import datetime
from scipy.signal import savgol_filter
from collections import defaultdict
from pathlib import Path
from enum import Enum
from concurrent.futures import ProcessPoolExecutor

from mosgim.geo.geomag import geo2mag
from mosgim.geo.geomag import geo2modip
from mosgim.utils.time_util import sec_of_day, sec_of_interval
from mosgim.loader.loader import LoaderTxt, LoaderHDF

sites = ['019b', '7odm', 'ab02', 'ab06', 'ab09', 'ab11', 'ab12', 'ab13',
         'ab15', 'ab17', 'ab21', 'ab27', 'ab33', 'ab35', 'ab37', 'ab41',
         'ab45', 'ab48', 'ab49', 'ac03', 'ac21', 'ac61', 'acor', 'acso',
         'adis', 'ahid', 'aira', 'ajac', 'albh', 'alg3', 'alic', 'alme',
         'alon', 'alrt', 'alth', 'amc2', 'ankr', 'antc', 'areq', 'artu',
         'aruc', 'asky', 'aspa', 'auck', 'badg', 'baie', 'bake', 'bald',
         'bamo', 'bara', 'barh', 'bcyi', 'bell', 'benn', 'berp', 'bilb',
         'bjco', 'bjfs', 'bjnm', 'bla1', 'bluf', 'bogi', 'bogt', 'braz',
         'brip', 'brst', 'brux', 'bshm', 'bucu', 'budp', 'bums', 'buri',
         'bzrg', 'cand', 'cant', 'capf', 'cas1', 'casc', 'ccj2', 'cedu',
         'chan', 'chiz', 'chpi', 'chti', 'chur', 'cihl', 'cjtr', 'ckis',
         'clrk', 'cmbl', 'cn00', 'cn04', 'cn09', 'cn13', 'cn20', 'cn22',
         'cn23', 'cn40', 'cnmr', 'coco', 'con2', 'cord', 'coyq', 'crao',
         'cusv', 'daej', 'dakr', 'dane', 'darw', 'dav1', 'devi', 'dgar',
         'dgjg', 'drao', 'dubo', 'ecsd', 'ela2', 'eur2', 'faa1', 'falk',
         'fall', 'ffmj', 'flin', 'flrs', 'func', 'g101', 'g107', 'g117',
         'g124', 'g201', 'g202', 'ganp', 'gisb', 'glps', 'gls1', 'gls2',
         'gls3', 'glsv', 'gmma', 'gode', 'guat', 'guax', 'harb', 'hces',
         'hdil', 'helg', 'hlfx', 'hmbg', 'hnlc', 'hob2', 'hofn', 'holm',
         'howe', 'hsmn', 'hueg', 'ibiz', 'iisc', 'ilsg', 'inmn', 'invk',
         'iqal', 'iqqe', 'irkj', 'isba', 'isco', 'ista', 'joen', 'karr',
         'kely', 'khar', 'khlr', 'kir0', 'kiri', 'kour', 'ksnb', 'ksu1',
         'kuuj', 'kvtx', 'lamp', 'laut', 'lcsb', 'lhaz', 'lkwy', 'lovj',
         'lply', 'lthw', 'mac1', 'mag0', 'mal2', 'mar6', 'marg', 'mas1',
         'mat1', 'maw1', 'mcar', 'mdvj', 'mkea', 'mobs', 'moiu', 'morp',
         'nain', 'naur', 'nium', 'nnor', 'noa1', 'not1', 'novm', 'nril',
         'nya1', 'ohi2', 'ons1', 'p008', 'p038', 'p050', 'p776', 'p778',
         'p803', 'palm', 'parc', 'park', 'pece', 'penc', 'pets', 'pimo',
         'pirt', 'pngm', 'pol2', 'pove', 'qaar', 'qaq1', 'qiki', 'recf',
         'reso', 'riop', 'rmbo', 'salu', 'sask', 'savo', 'sch2', 'scor',
         'sg27', 'sgoc', 'shao', 'soda', 'stas', 'stew', 'sthl', 'stj2',
         'sumk', 'suth', 'syog', 'tash', 'tehn', 'tetn', 'tixg', 'tomo',
         'tor2', 'tow2', 'trds', 'tro1', 'tuc2', 'tuva', 'udec', 'ufpr',
         'ulab', 'unbj', 'unpm', 'urum', 'usmx', 'vacs', 'vars', 'vis0',
         'vlns', 'whit', 'whng', 'whtm', 'will', 'wind', 'wway', 'xmis',
         'yakt', 'yell', 'ykro', 'ymer', 'zamb']

#sites = sites[22:28]

class DataSourceType(Enum):
    hdf = 'hdf'
    rinex = 'rinex'
    txt = 'txt'

    def __str__(self):
        return self.value
    
class MagneticCoordType(Enum):
    mag = 'mag'
    mdip = 'mdip'

    def __str__(self):
        return self.value
    
class ProcessingType(Enum):
    single = 'single'
    ranged = 'ranged'

    def __str__(self):
        return self.value

def process_data(data_generator):
    all_data = defaultdict(list)
    count = 0
    for data, data_id in data_generator:
        if data.shape==():
            print(f'No data for {data_id}')
            continue
        times = data['datetime'][:]
        data_days = [datetime(d.year, d.month, d.day) for d in times]
        if len(set(data_days)) != 1:
            msg = f'{data_id} is not processed: multiple days presented '
            msg += f'{set(data_days)}. Skip.'
            print(msg)
            continue
        try:
            prepared = process_intervals(data, maxgap=35., 
                                         maxjump=2., 
                                         derivative=False)
            count += len(prepared['dtec'])
            for k in prepared:
                all_data[k].extend(prepared[k])
        except Exception as e:
            print(f'{data_id} not processed. Reason: {e}')
    return all_data


def get_continuos_intervals(data, maxgap=30, maxjump=1):
    return getContInt(data['sec_of_day'][:], 
                      data['tec'][:], 
                      data['ipp_lon'][:], data['ipp_lat'][:], 
                      data['el'][:],  
                      maxgap=maxgap, maxjump=maxjump)


def getContInt(times, tec, lon, lat, el,  maxgap=30, maxjump=1):
    r = np.array(range(len(times)))
    idx = np.isfinite(tec) & np.isfinite(lon) & np.isfinite(lat) & np.isfinite(el) & (el > 10.)
    r = r[idx]
    intervals = []
    if len(r) == 0:
        return idx, intervals
    beginning = r[0]
    last = r[0]
    last_time = times[last]
    for i in r[1:]:
        if abs(times[i] - last_time) > maxgap or abs(tec[i] - tec[last]) > maxjump:
            intervals.append((beginning, last))
            beginning = i
        last = i
        last_time = times[last]
        if i == r[-1]:
            intervals.append((beginning, last))
    return idx, intervals

def process_intervals(data, maxgap, maxjump, derivative, 
                      short = 3600, sparse = 600):
    result = defaultdict(list)
    tt = sec_of_day(data['datetime'])        
    idx, intervals = getContInt(tt, 
                                data['tec'], data['ipp_lon'],
                                data['ipp_lat'], data['el'],  
                                maxgap=maxgap, maxjump=maxjump)
    #_, intervals = get_continuos_intervals(data, maxgap=maxgap, maxjump=maxjump)
    for start, fin in intervals:

        if (tt[fin] - tt[start]) < short:    # disgard all the arcs shorter than 1 hour
            #print('too short interval')
            continue
        ind_sparse = (tt[start:fin] % sparse == 0)
        data_sample = data[start:fin]
        data_sample['tec'] = savgol_filter(data_sample['tec'][:], 21, 2)
        data_sample = data_sample[ind_sparse]
        
        if derivative == True:
            dtec = data_sample['tec'][1:] - data_sample['tec'][0:-1]
            data_out = data_sample[1:]
            data_ref = data_sample[0:-1]
        
        if derivative == False:
            idx_min = np.argmin(data_sample['tec'])
            data0 = data_sample[idx_min]
            data_out = np.delete(data_sample, idx_min)
            dtec = data_out['tec'][:] - data0['tec']
            data_ref = np.zeros_like(data_out)
            data_ref[:] = data0
        result['dtec'].append(dtec)
        result['out'].append(data_out)
        result['ref'].append(data_ref)
    return result
    
def combine_data(all_data, nchunks=1):
    count = 0
    for arr in all_data['dtec']:
        count += arr.shape[0]
    tec_data = np.concatenate(tuple(o for o in all_data['dtec']))
    out_data = np.concatenate(tuple(o for o in all_data['out']))
    ref_data = np.concatenate(tuple(r for r in all_data['ref']))
    fields = ['tec', 'time', 'lon', 'lat', 'el', 'rtime', 'rlon', 'rlat', 'rel',
              'colat_mdip', 'mlt_mdip', 'rcolat_mdip', 'rmlt_mdip',
              'colat_mag', 'mlt_mag', 'rcolat_mag', 'rmlt_mag']
    combs = []
    ichunks = get_chunk_indexes(count, nchunks)
    for i, (start, fin) in enumerate(ichunks):
        print(start, fin)
        comb = dict()
        #fields = {k: i for i, k in enumerate(fields)}
        comb['tec'] = tec_data[start:fin]
        _out_data = out_data[start:fin]
        comb['time'] = _out_data['datetime']
        comb['lon'] = _out_data['ipp_lon']
        comb['lat'] = _out_data['ipp_lat']
        comb['el'] = _out_data['el']
        _ref_data = ref_data[start:fin]
        comb['rtime'] = _ref_data['datetime']
        comb['rlon'] = _ref_data['ipp_lon']
        comb['rlat'] = _ref_data['ipp_lat']
        comb['rel'] = _ref_data['el']
        for f in fields:
            if f in comb:
                continue
            comb[f] = np.zeros(comb['tec'].shape)
        combs.append(comb)
    return combs


def get_chunk_indexes(size, nchunks):
    if nchunks > 1:
        step = int(size / nchunks)
        ichunks = [(i-step, i) for i in range(step, size, step)]
        if (size - ichunks[-1][1]) / size > 0.1 * step:
            ichunks += [(ichunks[-1][1], size)]
        else:
            ichunks[-1] = (ichunks[-1][0], size)
        return ichunks
    elif nchunks <= 1:
        return [(0, size)]
    

def calc_mag(comb, g2m):
    colat, mlt = \
        g2m(np.pi/2 - rad(comb['lat']), rad(comb['lon']), comb['time'])  
    return colat, mlt

def calc_mag_ref(comb, g2m):
    rcolat, rmlt = \
        g2m(np.pi/2 - rad(comb['rlat']), rad(comb['rlon']), comb['rtime'])  
    return  rcolat, rmlt

def calc_mag_coordinates(comb):
    comb['colat_mdip'], comb['mlt_mdip'] = calc_mag(comb, geo2modip)  
    comb['rcolat_mdip'], comb['rmlt_mdip'] = calc_mag_ref(comb, geo2modip)
    comb['colat_mag'], comb['mlt_mag'] = calc_mag(comb, geo2mag)
    comb['rcolat_mag'], comb['rmlt_mag'] = calc_mag_ref(comb, geo2mag)
    return comb
    
def calculate_seed_mag_coordinates_parallel(chunks, nworkers=3):
    if len(chunks) < 1:
        return None
    if len(chunks) == 1:
        calc_mag_coordinates(chunks[0])
        return chunks[0]
    chunks_processed = []
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        queue = []
        for chunk in chunks:
            query = executor.submit(calc_mag_coordinates, chunk)
            queue.append(query)
        for v in concurrent.futures.as_completed(queue):
            chunk = v.result()
            chunks_processed.append(chunk)
    count = 0
    for chunk in chunks_processed:
        count += chunk['tec'].shape[0]
    comb = {}
    for f in chunks_processed[0]:
        if not f in ['time', 'rtime']:
            comb[f] = np.zeros((count,)) 
    comb['time'] = np.zeros((count,), dtype=object)
    comb['rtime'] = np.zeros((count,), dtype=object)
    start, end = 0, 0 
    for chunk in chunks_processed:
        end = end + chunk['tec'].shape[0]
        
        for k in chunk:
            comb[k][start:end] = chunk[k]
        start = end
    return comb
    

def save_data(comb, modip_file, mag_file, day_date):
    mags = [MagneticCoordType.mdip, MagneticCoordType.mag]
    for mtype, filename in zip(mags, [modip_file, mag_file]):
        data = get_data(comb, mtype, day_date)
        postf = str(mtype)
        np.savez(filename, 
                day = day_date,
                **data)    
        
def get_data(comb, mtype, day_date):
    postf = str(mtype)
    data = dict(time = sec_of_interval(comb['time'], day_date), 
                mlt = comb['mlt_' + postf ], 
                mcolat = comb['colat_' + postf], 
                el = rad(comb['el']),
                time_ref = sec_of_interval(comb['rtime'], day_date), 
                mlt_ref = comb['rmlt_' + postf], 
                mcolat_ref = comb['rcolat_' + postf], 
                el_ref = rad(comb['rel']), 
                rhs = comb['tec'])    
    return data
    
    

if __name__ == '__main__':
    import argparse
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
    if args.nsites:
        sites = sites[:args.nsites]
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
