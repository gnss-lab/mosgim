import os
import time
import numpy as np
import h5py
import concurrent.futures
from datetime import datetime
from warnings import warn
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


from geo import HM
from geo import sub_ionospheric


class Loader():
    
    def __init__(self):
        self.FIELDS = ['datetime', 'el', 'ipp_lat', 'ipp_lon', 'tec']
        self.DTYPE = (object, float, float, float, float)
        self.not_found_sites = []

class LoaderTxt(Loader):
    
    def __init__(self, root_dir):
        super().__init__()
        self.dformat = "%Y-%m-%dT%H:%M:%S"
        self.root_dir = root_dir

    def get_files(self, rootdir):
        result = defaultdict(list)
        for subdir, _, files in os.walk(rootdir):
            for filename in files:
                filepath = Path(subdir) / filename
                if str(filepath).endswith(".dat"):
                    site = filename[:4]
                    if site != subdir[-4:]:
                        raise ValueError(f'{site} in {subdir}. wrong site name')
                    result[site].append(filepath)
                else:
                    warn(f'{filepath} in {subdir} is not data file')
        for site in result:
            result[site].sort()
        return result

    def load_data(self, filepath):
        convert = lambda x: datetime.strptime(x.decode("utf-8"), self.dformat)
        data = np.genfromtxt(filepath, 
                             comments='#', 
                             names=self.FIELDS, 
                             dtype=self.DTYPE,
                             converters={"datetime": convert},  
                             #unpack=True
                             )

        #tt = sec_of_day(data['datetime'])
        #data = append_fields(data, 'sec_of_day', tt, np.float)
        return data, filepath
    
    def __load_data_pool(self, filepath):
        return self.load_data(filepath), filepath

    def generate_data(self, sites=[]):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        for site, site_files in files.items():
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            count = 0
            st = time.time()
            for sat_file in site_files:
                try:
                    data, _ = self.load_data(sat_file)
                    count += 1
                    yield data, sat_file
                except Exception as e:
                    print(f'{sat_file} not processed. Reason: {e}')
            print(f'{site} contribute {count} files, takes {time.time() - st}')
            
    def generate_data_pool(self, sites=[], nworkers=1):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            queue = []
            for site, site_files in files.items():
                if sites and not site in sites:
                    continue
                self.not_found_sites.remove(site)
                count = 0
                st = time.time()
                for sat_file in site_files:
                    try:
                        query = executor.submit(self.load_data, sat_file)
                        queue.append(query)
                        print(f'Added {sat_file} in queue')
                    except Exception as e:
                        print(f'{sat_file} not processed. Reason: {e}')
            for v in concurrent.futures.as_completed(queue):
                print
                yield v.result()


class LoaderHDF(Loader):
    
    def __init__(self, hdf_path):
        super().__init__()
        self.hdf_path = hdf_path
        self.hdf_file = h5py.File(hdf_path, 'r')
    
    def generate_data(self, sites=[]):
        self.not_found_sites = sites[:]
        for site in self.hdf_file:
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            slat = self.hdf_file[site].attrs['lat']
            slon = self.hdf_file[site].attrs['lon']
            st = time.time()
            count = 0
            for sat in self.hdf_file[site]:
                sat_data = self.hdf_file[site][sat]
                arr = np.empty((len(sat_data['tec']),), 
                               list(zip(self.FIELDS,self.DTYPE)))
                el = sat_data['elevation'][:]
                az = sat_data['azimuth'][:]
                ts = sat_data['timestamp'][:]
                ipp_lat, ipp_lon = sub_ionospheric(slat, slon, HM, az, el)
                
                arr['datetime'] = np.array([datetime.utcfromtimestamp(float(t)) for t in ts])
                arr['el'] = np.rad2deg(el)
                arr['ipp_lat'] = np.rad2deg(ipp_lat)
                arr['ipp_lon'] = np.rad2deg(ipp_lon)
                arr['tec'] = sat_data['tec'][:]
                count += 1
                yield arr, sat + '_' + site
            print(f'{site} contribute {count} files, takes {time.time() - st}')

                
                
