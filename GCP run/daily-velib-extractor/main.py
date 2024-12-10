import requests, datetime
import os
from sqlalchemy import MetaData
from sqlalchemy.sql import insert
from google.cloud.sql.connector import Connector
import pg8000
import sqlalchemy

def connect_to_cloud_sql():
    """
    Connect to the Cloud SQL instance using the Cloud SQL Python Connector.
    """
    try:
        # Get the connection details from environment variables
        db_user = "velib_daily"
        db_pass = SQL_PASSWORD
        db_name = "velib"
        connection_name = "dreamteam-406713:europe-west9:lucinia-sql"  # Connection name

        # Initialize the Cloud SQL connector
        connector = Connector()
        def getconn() -> pg8000.dbapi.Connection:
            conn: pg8000.dbapi.Connection = connector.connect(
                connection_name,
                "pg8000",
                user=db_user,
                password=db_pass,
                db=db_name,
            )
            return conn

        # Establish a secure connection to the Cloud SQL instance
        pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            # ...
        )

        print("Connected to the database successfully!")
        return pool

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
        # if station:
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
            'poll_dt' : datetime.datetime.now()
        }
        results.append(r)
    print(f'Loaded {len(results)} items.')
    return results

def push_results(p, results):
    metadata = MetaData()
    metadata.reflect(bind=p)
    velib_stations = metadata.tables['velib_status']
    insert_stmt = insert(velib_stations)
    print(f'Adding {len(results)} rows to "velib_status"')
    with p.begin() as transaction:
        c = transaction.execute(insert_stmt, results)
    print(f'SQL job finished.')
    return str(c.rowcount)

def main(req):
    debug = True if req.get_json().get('debug') else False
    results = get_velib_status(debug)
    return push_results(connect_to_cloud_sql(), results)

