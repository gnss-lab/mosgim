import numpy as np
import time

from numpy import deg2rad as rad
from datetime import datetime
from scipy.signal import savgol_filter
from collections import defaultdict
from pathlib import Path
from enum import Enum

from geomag import geo2mag
from geomag import geo2modip
from time_util import sec_of_day, sec_of_interval
from loader import LoaderTxt, LoaderHDF

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


class DataSourceType(Enum):
    hdf = 'hdf'
    rinex = 'rinex'
    txt = 'txt'

    def __str__(self):
        return self.value

def process_data(data_generator):
    all_data = defaultdict(list)
    count = 0
    for data, data_id in data_generator:
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
            data_ref = data_out[:]
            data_ref[:] = data0

        result['dtec'].append(dtec)
        result['out'].append(data_out)
        result['ref'].append(data_ref)
    return result
    
def combine_data(all_data):
    count = 0
    for arr in all_data['dtec']:
        count += arr.shape[0]
    out_data = np.concatenate(tuple(o for o in all_data['out']))
    ref_data = np.concatenate(tuple(r for r in all_data['ref']))
    fields = ['tec', 'time', 'lon', 'lat', 'el', 'rtime', 'rlon', 'rlat', 'rel',
              'colat_mdip', 'mlt_mdip', 'rcolat_mdip', 'rmlt_mdip',
              'colat_mag', 'mlt_mag', 'rcolat_mag', 'rmlt_mag']
    comb = {f: np.zeros((count,)) for f in fields}
    #fields = {k: i for i, k in enumerate(fields)}
    comb['tec'] = all_data['dtec'][:]
    comb['time'] = out_data['datetime'][:]
    comb['lon'] = out_data['ipp_lon'][:]
    comb['lat'] = out_data['ipp_lat'][:]
    comb['el'] = out_data['el'][:]
    comb['rtime'] = ref_data['datetime'][:]
    comb['rlon'] = ref_data['ipp_lon'][:]
    comb['rlat'] = ref_data['ipp_lat'][:]
    comb['rel'] = ref_data['el'][:]
    return comb

def calc_mag(comb, g2m):
    colat, mlt = \
        g2m(np.pi/2 - rad(comb['lat']), rad(comb['lon']), comb['time'])  
    return colat, mlt

def calc_mag_ref(comb, g2m):
    rcolat, rmlt = \
        g2m(np.pi/2 - rad(comb['rlat']), rad(comb['rlon']), comb['rtime'])  
    return  rcolat, rmlt

def seed_mag_coordinates(comb):
    comb['colat_mdip'], comb['mlt_mdip'] = calc_mag(comb, geo2modip)  
    comb['rcolat_mdip'], comb['rmlt_mdip'] = calc_mag_ref(comb, geo2modip)
    comb['colat_mag'], comb['mlt_mag'] = calc_mag(comb, geo2mag)
    comb['rcolat_mag'], comb['rmlt_mag'] = calc_mag_ref(comb, geo2mag)

def save_data(comb, modip_file, mag_file, day_date):
    for postf, filename in zip(['mdip', 'mag'], [modip_file, mag_file]):
        np.savez(filename, 
                day = day_date,
                time = sec_of_interval(comb['time'], day_date), 
                mlt = comb['mlt_' + postf ], 
                mcolat = comb['colat_' + postf], 
                el = rad(comb['el']),
                time_ref = sec_of_interval(comb['rtime'], day_date), 
                mlt_ref = comb['rmlt_' + postf], 
                mcolat_ref = comb['rcolat_' + postf], 
                el_ref = rad(comb['rel']), 
                rhs = comb['tec'])    

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
    args = parser.parse_args()
    if args.data_source == DataSourceType.hdf:
        loader = LoaderHDF(args.data_path)
        data_generator = loader.generate_data(sites=sites)
    if args.data_source == DataSourceType.txt:
        loader = LoaderTxt(args.data_path)
        process_date = args.date
        data_generator = loader.generate_data()
    data = process_data(data_generator)
    combined_data = combine_data(data)
    print('Start magnetic calculations...')
    st = time.time()
    seed_mag_coordinates(combined_data)
    print(f'Done, took {time.time() - st}')
    save_data(combined_data, args.modip_file, args.mag_file, process_date)
