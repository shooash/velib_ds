import psycopg2
import pandas as pd
from general import log

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
            log("Openning PostgreSQL connection.")
            conn = psycopg2.connect(host=self.DB_HOST, 
                                    database=self.DB_NAME, 
                                    user=self.DB_USER, 
                                    password=self.DB_PASSWORD)
            log("Running SQL query.")
            cur = conn.cursor()
            cur.execute(query)
            log("Fetching PostgreSQL data.")

            self.rows = cur.fetchall()
            self.cols = [c.name for c in cur.description] if cur.description is not None else []
            self.types = [self.__get_type(c.type_code) for c in cur.description] if cur.description is not None else []
        except (Exception, psycopg2.Error) as error:
            log(f"Error connecting to or querying the database: {error}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
                log("PostgreSQL connection closed.")

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