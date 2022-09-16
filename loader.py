import os
import time
import numpy as np
from datetime import datetime
from warnings import warn
from collections import defaultdict
from pathlib import Path


class Loader():
    
    def __init__(self):
        self.FIELDS = ['datetime', 'el', 'ipp_lat', 'ipp_lon', 'tec']
        self.DTYPE = (object, float, float, float, float)

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
                             unpack=True)
        #tt = sec_of_day(data['datetime'])
        #data = append_fields(data, 'sec_of_day', tt, np.float)
        return data

    def generate_data(self):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        for site, site_files in files.items():
            count = 0
            st = time.time()
            for sat_file in site_files:
                try:
                    data = self.load_data(sat_file)
                    count += 1
                    yield data, sat_file
                except Exception as e:
                    print(f'{sat_file} not processed. Reason: {e}')
            print(f'{site} contribute {count} files, takes {time.time() - st}')

