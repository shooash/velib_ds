import psycopg2, datetime, requests
import pandas as pd
from time import sleep
from io import StringIO
import requests

class VelibConnector:
    """
    Basic Postgre connector to load Velib historical data for Velib DataScientest project
    Usage examples:
        df = VelibConnector("SELECT * FROM velib_all WHERE dt > '2025-01-31'").to_pandas()
        _, cols = VelibConnector("SELECT * FROM velib_all WHERE 1<>1").raw()
        data_list, cols = VelibConnector("SELECT * FROM velib_all order by dt desc limit 100").to_list()
        data_dict = VelibConnector("SELECT * FROM velib_all WHERE dt < '2025-01-01'").to_dict()
    """
    DB_HOST = '34.163.111.108'
    DB_NAME = 'velib'
    DB_USER = 'reader'
    DB_PASSWORD = 'public'
    known_types = [
        psycopg2.NUMBER,
        psycopg2.STRING,
        psycopg2.DATETIME,
        psycopg2.BINARY
    ]

    def __init__(self, query="SELECT * FROM velib_all limit 10", debuguser = None):
        if debuguser:
            self.DB_USER = debuguser['user']
            self.DB_PASSWORD = debuguser['pass']
        try:
            print("Openning PostgreSQL connection.")
            conn = psycopg2.connect(host=self.DB_HOST, 
                                    database=self.DB_NAME, 
                                    user=self.DB_USER, 
                                    password=self.DB_PASSWORD)
            cur = conn.cursor()
            cur.execute(query)
            print("Fetching PostgreSQL data.")

            self.rows = cur.fetchall()
            self.cols = [c.name for c in cur.description]
            self.types = [self.__get_type(c.type_code) for c in cur.description]
        except (Exception, psycopg2.Error) as error:
            print(f"Error connecting to or querying the database: {error}")
        finally:
            if conn:
                cur.close()
                conn.close()
                print("PostgreSQL connection closed.")
    def __get_type(self, key):
        for t in self.known_types:
            if key in t.values:
                return t.name

    def to_pandas(self):
        return pd.DataFrame(self.rows, columns=self.cols)
    def raw(self):
        return self.rows, self.cols
    def to_list(self): 
        return list(zip(*self.rows)), self.cols
    def to_dict(self):
        return {name: list(column) for name, column in zip(self.cols, zip(*self.rows))}


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
    def __init__(self, from_dt : datetime.datetime = datetime.datetime(2024, 12, 6, 0, 0, 0), to_dt : datetime.datetime = None, station : str = None):
        if station is None:
            station = self.SELECTED_STATION
        if to_dt is None:
            to_dt = datetime.date.today()
        from_dt_str = from_dt.strftime('%Y-%m-%dT%H:00:00Z')
        to_dt_str = to_dt.strftime('%Y-%m-%dT%H:00:00Z')
        url = self.DATA_NODE.format(station = station, from_dt = from_dt_str, to_dt = to_dt_str)
        head = {
            'accept' : '*/*',
            'apikey' : self.APIKEY
        }
        print(f'Chargement de données MétéoFrance pour la période: {from_dt_str} - {to_dt_str}')
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