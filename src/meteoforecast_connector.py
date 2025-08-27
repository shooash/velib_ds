import pandas as pd
import pytz
import requests
import datetime
from general import log

class OpenWeatherConnector:
    """
    Usage: forecast_data = OpenWeatherConnector(
            locations = [{'station' : 'stationid', 'lat' : 48.857992591527, 'lon' : 2.3469167947769 }]
            ).to_pandas()
    """
    APIKEY = '2e26431ef9352f60634c1bc003c2c9ce'
    FORECAST_NODE = "https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={key}&units=metric"

    def __init__(self, locations : list[dict]):
        self.data = []
        for station in locations:
            url = self.FORECAST_NODE.format(lat=station['lat'], lon=station['lon'], key=self.APIKEY)
            log(f"""Chargement de données OpenWeather pour la station: {station['station']}""")
            try:
                q = requests.get(url)
            except Exception as e:
                raise e
            if q.status_code != 200:
                raise Exception(f'Failed to load OpenWeather data: code {q.status_code}, details: {q.text}')
            result = q.json()
            # La liste 'list' contient les données par 3 heures.
            for r in result['list']:
                rain3h = r['rain']['3h'] if 'rain' in r else 0
                snow3h = r['snow']['3h'] if 'snow' in r else 0
                temp = r['main']['temp']
                gel = 0 if (temp >= 0) else min(20 * -temp, 60)
                item = {
                    'station' : station['station'],
                    'datehour' : datetime.datetime.fromtimestamp(r['dt'], datetime.timezone.utc).astimezone(pytz.timezone('Europe/Paris')).replace(tzinfo=None),
                    'temp' : temp,
                    'precip' : rain3h + snow3h,
                    'gel' : gel,
                    'vent' : r['wind']['speed'] if 'wind' in r else 0
                }
                self.data.append(item)

    def to_pandas(self):
        df = pd.DataFrame.from_records(self.data)
        # return df
        stations = df[['station']].drop_duplicates()
        ### Convert original UTC dt to Paris
        df['datehour'] = df['datehour'].dt.tz_localize(tz='UTC').dt.tz_convert(tz='Europe/Paris').dt.tz_localize(None)
        seasonality = pd.DataFrame({'datehour' : pd.date_range(df.datehour.min() - datetime.timedelta(hours=2), df.datehour.max() + datetime.timedelta(hours=1), freq='h', tz='Europe/Paris').tz_convert(None)})
        seasonality_stations = stations.merge(seasonality, how='cross')
        df = seasonality_stations.merge(df, how='left', on=['station', 'datehour'])
        # df = df.merge(right=seasonality, how='outer', on='datehour')
        # Lissage de données de précipitation - ils étaient fournis en somme de 3 heures
        df['precip'] = df['precip'].fillna(0)
        df['precip'] = df.groupby('station')['precip'].transform(lambda x: x.rolling(window=3, center=True, min_periods=1).mean())
        # df['precip'] = df['precip'].fillna(0).groupby('station').rolling(window=3, center=True, min_periods=1).mean()
        # Interpolation linéaire d'autres données
        df = df.interpolate()
        df = df.bfill()
        return df
    
    def raw(self):
        return self.data
    
    def to_list(self):
        return [list(v.values()) for v in self.data]
    def to_dict(self):
        return {name: list(column) for name, column in zip(self.data[0].keys(), zip(*[list(v.values()) for v in self.data]))}
    def cols(self):
        return list(zip(*[list(v.values()) for v in self.data])), list(self.data[0].keys())
