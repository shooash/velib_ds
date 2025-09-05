import requests, datetime
import os
import psycopg2
from psycopg2.extras import execute_values

def connect_to_cloud_sql():
    """
    Connect to the Cloud SQL instance using the Cloud SQL Python Connector.
    """
    try:
        DB_USER = "velib_daily"
        DB_PASS = SQL_PASSWORD
        DB_NAME = "velib"
        DB_HOST = SQL_HOST

        conn = psycopg2.connect(host=DB_HOST, 
                                database=DB_NAME, 
                                user=DB_USER, 
                                password=DB_PASS)
        conn.autocommit = True
        cur = conn.cursor()
        print("Connected to the database successfully!")
        return cur
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise

def get_velib_status(debug = False):
    status_point = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
    data = requests.get(status_point)
    info = data.json()
    if debug:
        print(info, status_point)
    print(f"Fetched {len(info['data']['stations'])} stations data from {status_point}")
    results = []
    def get_type_count(l : list | None, t : str):
        if not l:
            return None
        for i in l:
            result = i.get(t)
            if result is not None:
                return result
        return None

    for station in info['data']['stations']:
        r = {
            'station' : station.get('station_id'),
            'bikes' : station.get('num_bikes_available'),
            'max_bikes' : station.get('num_docks_available'),
            'mechanical' : get_type_count(station.get('num_bikes_available_types'), 'mechanical'),
            'ebike' : get_type_count(station.get('num_bikes_available_types'), 'ebike'),
            'is_installed' : station.get('is_installed'),
            'is_returning' : station.get('is_returning'),
            'is_renting' : station.get('is_renting'),
            'dt' :  datetime.datetime.fromtimestamp(station.get('last_reported')),
            'poll_dt' : datetime.datetime.now(datetime.timezone.utc)
        }
        results.append(
            (
                r['station'], r['bikes'], r['max_bikes'],
                r['mechanical'], r['ebike'], r['is_installed'],
                r['is_returning'], r['is_renting'],
                r['dt'], r['poll_dt']
            )
        )
    print(f'Loaded {len(results)} items.')
    return results

INSERT_PRE = """
    INSERT INTO velib_status (
        station, bikes, max_bikes, mechanical, ebike,
        is_installed, is_returning, is_renting, dt, poll_dt
    )
    VALUES %s
"""
def push_results(cur, results):
    print(f'Adding {len(results)} rows to "velib_status"')
    execute_values(
        cur,
        INSERT_PRE,
        results)
    print(f'SQL job finished.')
    return str("Success")

def main(req):
    debug = True if req.get_json().get('debug') else False
    results = get_velib_status(debug)
    return push_results(connect_to_cloud_sql(), results)

