import datetime
from io import StringIO
from time import sleep
import pandas as pd
import pytz
import requests
from general import log

class MeteoFranceConnector:
    """
    Usage: meteo_data = MeteoFranceConnector(
        from_dt = datetime.datetime.strptime('2025-03-01', '%Y-%m-%d'), #optional, default 2024-12-06
        to_dt = datetime.datetime.strptime('2025-03-07', '%Y-%m-%d'), #optional, default today
        station = '75106001', #optional, meteo station, default MeteoFranceConnector.SELECTED_STATION
            ).to_pandas()
    """
    APIKEY = 'eyJ4NXQiOiJZV0kxTTJZNE1qWTNOemsyTkRZeU5XTTRPV014TXpjek1UVmhNbU14T1RSa09ETXlOVEE0Tnc9PSIsImtpZCI6ImdhdGV3YXlfY2VydGlmaWNhdGVfYWxpYXMiLCJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJzaG9vYXNoQGNhcmJvbi5zdXBlciIsImFwcGxpY2F0aW9uIjp7Im93bmVyIjoic2hvb2FzaCIsInRpZXJRdW90YVR5cGUiOm51bGwsInRpZXIiOiJVbmxpbWl0ZWQiLCJuYW1lIjoiRGVmYXVsdEFwcGxpY2F0aW9uIiwiaWQiOjI1MjU0LCJ1dWlkIjoiNmFiZjhiMzQtZmQ5NC00ZDg4LTk4MDUtZDM0MjFlZjRkMGUwIn0sImlzcyI6Imh0dHBzOlwvXC9wb3J0YWlsLWFwaS5tZXRlb2ZyYW5jZS5mcjo0NDNcL29hdXRoMlwvdG9rZW4iLCJ0aWVySW5mbyI6eyI1MFBlck1pbiI6eyJ0aWVyUXVvdGFUeXBlIjoicmVxdWVzdENvdW50IiwiZ3JhcGhRTE1heENvbXBsZXhpdHkiOjAsImdyYXBoUUxNYXhEZXB0aCI6MCwic3RvcE9uUXVvdGFSZWFjaCI6dHJ1ZSwic3Bpa2VBcnJlc3RMaW1pdCI6MCwic3Bpa2VBcnJlc3RVbml0Ijoic2VjIn19LCJrZXl0eXBlIjoiUFJPRFVDVElPTiIsInN1YnNjcmliZWRBUElzIjpbeyJzdWJzY3JpYmVyVGVuYW50RG9tYWluIjoiY2FyYm9uLnN1cGVyIiwibmFtZSI6IkRvbm5lZXNQdWJsaXF1ZXNDbGltYXRvbG9naWUiLCJjb250ZXh0IjoiXC9wdWJsaWNcL0RQQ2xpbVwvdjEiLCJwdWJsaXNoZXIiOiJhZG1pbl9tZiIsInZlcnNpb24iOiJ2MSIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9XSwiZXhwIjoxNzcyODc1OTg1LCJ0b2tlbl90eXBlIjoiYXBpS2V5IiwiaWF0IjoxNzQxMzM5OTg1LCJqdGkiOiI2ZjlhMzQ4My0wYjg1LTQyZGQtYjcyMy1kYjQxOWY2Mjk2MzgifQ==.ei8eq99GBwRGcBbStkbS0RDMmbnBg_y4xqZ3XIoau76ioF2BWi4b_TpI9XARuDwhEM8Y4829LjgxGHFgptPShA8QXX-QlnC3CqPGgN7x6niXF3JcJbEz0323lT5RCghqxr2ctg3OjfmYkdyJkBuHGsCUb1D7ywyH-dhTe8xdSsGk3AyIddcHj-AEEch6apjcbUmAbe-HWgdvBkMQWnCyWJypuAvMWNUW0SnMFcFkkJ7jLTNQ22M-sWTReTdjugM4TVQvWOhma8R6I6RjT4UMd_NPBj2QDgykUXK8DunC-9dFoU5Lmg152f4z0huk-vkLXfDKbGSSMOpT3psjd6dzzg=='
    STATIONS_NODE = 'https://public-api.meteofrance.fr/public/DPClim/v1/liste-stations/horaire?id-departement=75&parametre=temperature'
    DATA_NODE = 'https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/horaire?id-station={station}&date-deb-periode={from_dt}&date-fin-periode={to_dt}'
    FILE_NODE = 'https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={file_id}'
    # SELECTED_STATION = '75106001' #Station météo Luxembourg Paris
    SELECTED_STATION = '75116008' # Station hippodrome de Longchamp
    def __init__(self, from_dt : datetime.datetime = datetime.datetime(2024, 12, 6, 0, 0, 0, tzinfo=pytz.timezone('Europe/Paris')), to_dt : datetime.datetime | None = None, station : str | None = None):
        if station is None:
            station = self.SELECTED_STATION
        if to_dt is None:
            to_dt = datetime.datetime.now(tzinfo=pytz.timezone('Europe/Paris')) - datetime.timedelta(minutes=15)
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=pytz.timezone('Europe/Paris'))
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=pytz.timezone('Europe/Paris'))
        from_dt_str = from_dt.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:00:00Z')
        to_dt_str = to_dt.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        url = self.DATA_NODE.format(station = station, from_dt = from_dt_str, to_dt = to_dt_str)
        head = {
            'accept' : '*/*',
            'apikey' : self.APIKEY
        }
        log(f'Chargement de données MétéoFrance pour la période: {from_dt_str} - {to_dt_str} UTC')
        try:
            q = requests.get(url, headers=head)
        except Exception as e:
            raise e
        if q.status_code != 202:
            raise Exception(f'Failed to load meteo data: code {q.status_code}, details: {q.text}')
        result = q.json()
        file_id = result.get('elaboreProduitAvecDemandeResponse', {}).get('return')
        if not file_id:
            raise Exception(f'Failed to load meteo data: no file id provided.')
        sleep(2)
        url = self.FILE_NODE.format(file_id=file_id)
        try:
            q = requests.get(url, headers=head)
        except Exception as e:
            raise e
        if q.status_code != 201:
            raise Exception(f'Failed to load file for meteo data: file_id {file_id}, code {q.status_code}, details: {q.text}')
        self.data = q.text

    def to_pandas(self):
        return pd.read_csv(StringIO(self.data), index_col=False, sep=';')
    def raw(self):
        data_rows = self.data.splitlines()
        data_cols = data_rows[0].split(';')
        return [ r.split(';') for r in data_rows[1:]], data_cols
    def to_list(self): 
        data_rows = self.data.splitlines()
        data_cols = data_rows[0].split(';')
        return list(zip(*data_rows)), data_cols
    def to_dict(self):
        data_rows = self.data.splitlines()
        data_cols = data_rows[0].split(';')
        return {name: list(column) for name, column in zip(data_cols, zip(*data_rows))}
    def cols(self):
        data_rows = self.data.splitlines()
        return data_rows[0].split(';')