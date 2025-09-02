import os
# if os.getenv('USE_CUDA'):
#     import cudf.pandas
#     cudf.pandas.install()

from fastapi import HTTPException
import pandas as pd
import datetime
from typing import Dict, List, Union

import pytz
from velibdata import VelibDataDefaults, VelibData, save_dataset_stats, add_datetime_details
from transform import VelibTransformer
from meteoforecast_connector import OpenWeatherConnector
from general import DataFiles, log, Storage, load_transformer_params, load_best_params
from mlpflow import MLPFlow
from velibstations_connector import VelibStationsConnector

class FastAPIConnector:
    def __init__(self, model_name: str = "best_mlp", run_id: str = "best"):
        """
        Initialize the connector with the model.

        Args:
            model_name (str): Name of the model. Defaults to "best_mlp".
            run_id (str): MLflow run ID. Defaults to "best".
        """
        self.__df = None
        self.__df_full = None
        self.__df_stations = None

    @property
    def df(self):
        """Load dataframe from file or cache"""
        if self.__df is None:
            self.__df = pd.read_hdf(DataFiles.processed_7d, key='df')
        return self.__df.copy()

    @property
    def df_full(self):
        """Load dataframe from file or cache"""
        if self.__df_full is None:
            self.__df_full = pd.read_hdf(DataFiles.processed, key='df')
        return self.__df_full.copy()

    @property
    def df_stations(self):
        """Load dataframe from file or cache"""
        if self.__df_stations is None:
            log(f'Loading data from {DataFiles.processed_stations}...')
            self.__df_stations = pd.read_hdf(DataFiles.processed_stations, key='df').astype({'station': str, 'lat' : float, 'lon' : float, 'capacity' : float})
            log('Data loaded.')
        return self.__df_stations.copy()


    def get_last_timestamp(self):
        """Get last known record datehours"""
        return self.df['datehour'].max()
    
    def add_meteo_hist(self, df : pd.DataFrame):
        """Add historical weather data to dataframe"""
        # Filter columns to avoid duplicates
        meteo_columns = ["temp", "precip", "gel", "vent"]
        filtered_columns = [c for c in df.columns if c not in meteo_columns]
        return df[filtered_columns].merge(self.df[['datehour', 'station'] + meteo_columns], on=['datehour', 'station'])
        
    def add_meteo_future(self, df : pd.DataFrame):
        """Add predicted weather data to dataframe"""
        # Filter columns to avoid duplicates
        meteo_columns = ["temp", "precip", "gel", "vent"]
        filtered_columns = [c for c in df.columns if c not in meteo_columns]
        # locations = df[['station', 'lat', 'lon']].drop_duplicates().to_dict(orient='records')
        # Load only one station data - the one used for training
        locations = [{
            "station" : "Hippodrome de Longchamp",
            "lat": 48.85483333333333,
            "lon": 2.2336666666666667,
        }]
        log(f'Loading OpenWeather forecasts for {len(locations)} locations...')
        forecast_data = OpenWeatherConnector(locations).to_pandas()
        return_df = df[filtered_columns].merge(forecast_data[['datehour'] + meteo_columns], on=['datehour'], how='left')
        return return_df
    
    def get_stations_df(self) -> pd.DataFrame:
        connector = VelibStationsConnector()
        return connector.to_pandas()
        
    def get_stations(self) -> dict:
        try:
            stations_df = self.df_stations
            return {'stations' : stations_df.to_dict(orient='records')}
        except Exception as e:
            raise HTTPException(500, f"Failed to get stations: {str(e)}")

    def add_meteo(self, df : pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['historical'] = df['datehour'] <= self.get_last_timestamp()
        if df['historical'].all():
            return self.add_meteo_hist(df.drop(columns='historical'))
        if not df['historical'].any():
            return self.add_meteo_future(df.drop(columns='historical'))
        return pd.concat([self.add_meteo_hist(df[df['historical']]), self.add_meteo_future(df[~df['historical']])]).drop(columns='historical')

        
    def predict(self, stations: str | list[str], datehour: str | list[str]) -> Dict[str, Union[str, float]]:
        """
        Predict bike delta for a specific station and date/time.

        Args:
            station (str): Vélib station identifier (e.g., "82328045").
            date (str): Date/time in format "YYYY-MM-DD HH:MM".

        Returns:
            Dict: Prediction result with station, datehour, and predicted delta.
        """
        try:
            log(f'Running prediction for {len(stations)} stations at {len(datehour)} timestamps')
            # We need to construct the following model:
            # ['datehour', 'date', 'station', 'lat', 'lon', 'name', 'hour', 'month',
            # 'weekday', 'weekend', 'holiday', 'preholiday', 'postholiday', 'pont',
            # 'vacances', 'vacances_uni', 'reconstructed', 'capacity',
            # 'temp', 'precip', 'gel', 'vent']
            if isinstance(stations, str):
                stations = [stations]
            if isinstance(datehour, str):
                datehour = [datehour]
            date_list = []
            results_date_list = []
            station_list = []
            for s in stations:
                for d in datehour:
                    station_list.append(s)
                    date_list.append(pd.Timestamp(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M")).floor('h'))
                    results_date_list.append(d)
            df = pd.DataFrame({'datehour' : date_list, 'station' : station_list})
            # Add stations static data
            stations_df = self.df_stations
            df = df.merge(stations_df[['station', 'lat', 'lon', 'capacity']], on='station', how='left')
            df = add_datetime_details(df)
            # Add meteo data
            df = self.add_meteo(df)
            # reconstructed column
            df['reconstructed'] = False
            # target empty
            df['delta'] = 0
            # Load and apply transformer
            if df.isna().any().any():
                log(f'Warning: missing data after weather processing for {df.isna().sum().max()} of {len(df)} rows. Are the dates too far in the future? Clearing.')
                df = df.dropna()
            transformer = VelibTransformer.load(DataFiles.transformer)
            # transformer.params['reconstructed'] = True # No need for cleanup
            df = transformer.transform(df)
            model = MLPFlow.load('best')
            df['prediction'] = model.predict(df.drop(columns='delta'))
            model.end()
            return {
                "station": df['station'].astype(str).to_list(),
                "datehour": df['datehour'].dt.strftime("%Y-%m-%d %H:%M").to_list(),
                "prediction": df['prediction'].astype(float).to_list()
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to predict: {str(e)}")

    def predict_day(self, date: str | list[str]) -> Dict[str, Union[str, List[Dict]]]:
        """
        Predict bike delta for all stations for a given day.

        Args:
            date (str): Date in format "YYYY-MM-DD".

        Returns:
            Dict: Predictions for all stations for the day.
        """
        try:
            if isinstance(date, str):
                date = [date]
            stations_df = self.df_stations
            stations_list = stations_df['station'].astype(str).tolist()
            datehour_col = []
            for d in date:
                date_obj = datetime.datetime.strptime(d, "%Y-%m-%d")
                datehour_list = pd.date_range(start=date_obj, end=date_obj + datetime.timedelta(days=1) - datetime.timedelta(hours=1), freq='H').strftime("%Y-%m-%d %H:%M").tolist()
                datehour_col.extend(datehour_list)
            return self.predict(stations=stations_list, datehour=datehour_col)
        
        except Exception as e:
            raise HTTPException(500, f"Failed to predict day: {str(e)}")

    def predict_station_day(self, stations : str | list[str], date: str | list[str]) -> Dict[str, Union[str, List[Dict]]]:
        """

        Args:
            station (str | list[str]): Station name(s).
            date (str | list[str]): Date(s) in format "YYYY-MM-DD".

        Returns:
            Dict: Predictions for the specified station(s) and date(s).
        """
        try:
            if isinstance(stations, str):
                stations = [stations]
            if isinstance(date, str):
                date = [date]
            stations = list(filter(lambda s: s in self.df_stations['station'].values, stations))
            if not stations:
                raise Exception(f"No valid stations provided: {stations}")
            datehour_col = []
            for d in date:
                date_obj = datetime.datetime.strptime(d, "%Y-%m-%d")
                datehour_list = pd.date_range(start=date_obj, end=date_obj + datetime.timedelta(days=1) - datetime.timedelta(hours=1), freq='H').strftime("%Y-%m-%d %H:%M").tolist()
                datehour_col.extend(datehour_list)
            return self.predict(stations=stations, datehour=datehour_col)
        except Exception as e:
            raise HTTPException(500, f"Failed to predict station day: {str(e)}")

    def retrain(self) -> Dict[str, str]:
        """
        Retrain the model with full dataset from processed.h5.

        Returns:
            Dict: Status of the retraining process.
        """
        try:
            df = self.df_full
            transformer_params = load_transformer_params()
            if transformer_params.get('smoothen'):
                df = VelibTransformer.smoothen(df, transformer_params.get('smooth_window', 3))
            # Fit and save VelibTransformer
            transformer = VelibTransformer(transformer_params)
            df_transformed = transformer.fit_transform(df)
            # Save stations
            log(f'Saving stations with clusters to {DataFiles.processed_stations}.')
            stations_df = df[['station', 'lat', 'lon', 'name', 'capacity']].drop_duplicates()
            stations_df = transformer.clusterizer.transform(stations_df)
            stations_df.to_hdf(DataFiles.processed_stations, key='df', mode='w')

            log(f'Saving transformer to {DataFiles.transformer}')
            transformer.save(DataFiles.transformer)
            log('Creating and training MLPFlow model')
            model = MLPFlow()
            model.set_params(**load_best_params())
            model.fit(df_transformed.drop(columns='delta'), df_transformed['delta'])
            model.save('best')
            return {
                "status": "Model retrained successfully",
                "details": f"Trained on {len(df_transformed)} rows, val_loss: {min(model.history.history.get('val_loss', ['unknown']))}"
            }
        except Exception as e:
            raise
            raise HTTPException(500, f"Failed to retrain model: {str(e)}")


    def refresh(self) -> Dict[str, str]:
        """
        Refresh the datasets.
        Returns:
            Dict: Status of data refresh
        """
        try:
            # If dataset exists, load last timestamp for delta extraction
            is_delta = os.path.isfile(DataFiles.raw_velib) and os.path.isfile(DataFiles.raw_meteofrance)
            if is_delta:
                meteo_data = pd.read_hdf(DataFiles.raw_meteofrance, key='meteo')
                velib_data = pd.read_hdf(DataFiles.raw_velib, key='velib')
                velib_data['dt'] = pd.to_datetime(velib_data['dt']).dt.tz_localize(tz='Europe/Paris', nonexistent='shift_forward')
                from_dt : datetime.datetime = velib_data['dt'].max() # TZ aware for queries
                velib_data['dt'] = velib_data['dt'].dt.tz_localize(None) # TZ naive for post extraction
                log(f"Loading data since {from_dt.strftime('%d/%m/%Y, %H:%M:%S')} to be appended.")
            else:
                from_dt : datetime.datetime = VelibDataDefaults.from_dt
                log(f"Loading full dataset since {from_dt.strftime('%d/%m/%Y, %H:%M:%S')}.")
                
            to_dt=datetime.datetime.now(tz=pytz.timezone('Europe/Paris')) - datetime.timedelta(minutes=15) # Last available time for MeteoFrance
            # Exctraction
            data = VelibData(cache=False, from_dt=from_dt, to_dt=to_dt).extract()
            # Join datasets
            data_size = len(data.velib_data)
            log(f"Data extracted: {data_size} new rows.")
            if is_delta:
                data.velib_data = pd.concat([velib_data, data.velib_data])
                data.meteo_data = pd.concat([meteo_data, data.meteo_data])
            log(f"Saving velib data to {DataFiles.raw_velib}.")
            data.velib_data.to_hdf(DataFiles.raw_velib, key='velib', mode='w')
            log(f"Saving velib data to {DataFiles.raw_velib}.")
            data.meteo_data.to_hdf(DataFiles.raw_meteofrance, key='meteo', mode='w')

            total_data_size = len(data.velib_data)
            # Transform full set
            data.transform()
            if data.data is None:
                log('Got None for dataset from VelibData.')
                raise RuntimeError('Got None for dataset from VelibData.')
            # Don't save stats for prediction dataset
            save_dataset_stats(data.data)
            # Save processed dataset
            log(f"Saving processed data to {DataFiles.processed}.")
            data.data.to_hdf(DataFiles.processed, key='df', mode='w')
            # Save short version = last 7 days
            last_timestamp = data.data['datehour'].max()
            cutoff_7days = last_timestamp - datetime.timedelta(days=7)
            data.data[data.data['datehour'] >= cutoff_7days].to_hdf(DataFiles.processed_7d, key='df', mode='w')            
            log(f"Data refresh completed.")
            return {"status": "Data loaded successfully.",
                    "details": f"Fetched {data_size} rows to reach a total of {total_data_size} rows of raw data."}
        except Exception as e:
            log(f"Data refresh failed: {str(e)}.")
            raise HTTPException(500, f"Failed to refresh data: {str(e)}")


        