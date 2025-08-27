

import pandas as pd
import requests


class VelibStationsConnector:
    node = r'https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json'
    def __init__(self):
        try:
            q = requests.get(self.node)
        except Exception as e:
            raise e
        if q.status_code > 202:
            raise Exception(f'Failed to load file for stations data: code {q.status_code}, details: {q.text}')
        self.data = q.json().get('data', {}).get('stations')
        if not self.data:
            raise Exception(f'Empty data for stations.')
    def to_pandas(self):
        return pd.DataFrame(self.to_dict())
    def to_list(self): 
        return list(self.to_dict().values()), self.cols()
    def to_dict(self):
        keys = set().union(*self.data)
        default = dict.fromkeys(keys, None)
        full_data = [default | d for d in self.data]
        full_data = dict(zip(keys, zip(*[d.values() for d in full_data])))
        return {
            'station' : list(full_data['station_id']), 
            'lat' : list(full_data['lat']), 
            'lon' : list(full_data['lon']), 
            'name' : list(full_data['name']),
            'capacity' : list(full_data['capacity'])}
    def cols(self):
        return ['station', 'lat', 'lon', 'name', 'capacity']
