import datetime
import json
from typing import Union
import numpy as np
import pandas as pd
from pandas import notna
import pytz
from scipy.stats import zscore, ks_2samp as kstest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from velib_connector import VelibConnector
from meteo_connector import MeteoFranceConnector
from general import DataFiles, Storage, log

# pd.set_option('future.no_silent_downcasting', True)

class VelibDataDefaults:
    # Paramètres par defaut pour VelibData
    params = {
        'drop_stations_outliers' : True, # Enlever les stations peu représentées?
        'drop_days_outliers' : True, # Enlever les jours avec des données très limitées
        'reconstruct_velib' : True, # Reconstruire les données manquantes ?
        'velib' : True, # Charger les données vélib
        'meteo' : True, # Charger les donnéers météo
    }
    velib_store = DataFiles.raw_velib
    meteo_store = DataFiles.raw_meteofrance
    impute = False
    debug = True # Logger les erreurs
    cache = False # Charger les données du cache ou de GCP
    update_cache = False # Sauvegarder les données dans le cache
    from_dt = datetime.datetime(2024, 12, 5, 0, 0, 0, tzinfo=pytz.timezone('Europe/Paris')) # Date min
    to_dt = datetime.datetime.now(tz=pytz.timezone('Europe/Paris')) - datetime.timedelta(hours=1) # Date max

class VelibData:
    # Class ETL principal
    def __init__(self, 
                from_dt : datetime.datetime = VelibDataDefaults.from_dt, 
                to_dt : datetime.datetime = VelibDataDefaults.to_dt, 
                debug = VelibDataDefaults.debug, 
                params : dict = {},
                cache = VelibDataDefaults.cache,
                update_cache = VelibDataDefaults.update_cache):
        '''
        Data loader for DataScientest Velib project.
        from_dt, to_dt: min and max (excluded) dates for the data to be loaded. 
        The dataframe would hold information for timestamps > from_dt and timestamps < to_dt.
        '''
        if to_dt is None:
            to_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(0,0,0), tzinfo=pytz.timezone('Europe/Paris'))
        if from_dt.tzinfo is None:
            from_dt.astimezone(tz=pytz.timezone('Europe/Paris'))
        if to_dt.tzinfo is None:
            to_dt.astimezone(tz=pytz.timezone('Europe/Paris'))
        self.params = VelibDataDefaults.params | params
        self.from_dt : datetime.datetime = from_dt
        self.to_dt : datetime.datetime = to_dt
        self.debug = debug
        self.cache = cache
        self.update_cache = update_cache
        self.velib_store = VelibDataDefaults.velib_store
        self.meteo_store = VelibDataDefaults.meteo_store
    
    def extract(self):
        '''
        Extract all selected data from sources.
        '''
        if self.params['velib']:
            self.extract_velib()
        if self.params['meteo']:
            self.extract_meteo()
        return self

    def extract_velib(self):
        '''
        Extract velib data from GCP source. Consider dates as Europe/Paris.
        Return velib_data with tznaive dt
        '''
        if self.cache and not self.update_cache and Path(self.velib_store).is_file():
            # self.velib_data = pd.read_csv(self.velib_csv, parse_dates=['dt'])
            self.velib_data = pd.read_hdf(self.velib_store, key='velib')
            self.velib_data['dt'] = pd.to_datetime(self.velib_data['dt']).dt.tz_localize(tz='Europe/Paris', nonexistent='shift_forward').dt.tz_localize(None)
            return self
        cmd = f"""
        SELECT status_id, station, lat, lon,
            0 as delta, bikes, capacity, name,
                dt from velib_all
                where dt > '{self.from_dt.strftime(r'%Y%m%d %H:00:00')}' {"and dt <'" + self.to_dt.strftime(r'%Y%m%d %H:00:00') + "'" if self.to_dt else ""}
        """
        self.velib_data = VelibConnector(cmd).to_pandas()
        self.velib_data = self.fix_types(self.velib_data)
        if self.debug:
            log(f'{len(self.velib_data)} lignes chargées sur velib_data pour la période de {self.velib_data.dt.min()} à {self.velib_data.dt.max()}')     
        self.velib_data['dt'] = pd.to_datetime(self.velib_data['dt']).dt.tz_localize(tz='Europe/Paris', nonexistent='shift_forward').dt.tz_localize(None)
        if self.cache or self.update_cache:
            # self.velib_data.to_csv(self.velib_csv)
            self.velib_data.to_hdf(self.velib_store, key='velib', mode='w', complevel=7)
        return self

    def extract_meteo(self):
        '''
        Extract data from MeteoFrance API.
        '''
        if self.cache and not self.update_cache and Path(self.meteo_store).is_file():
            # self.meteo_data = pd.read_csv(self.meteo_csv)
            self.meteo_data = pd.read_hdf(self.meteo_store, key='meteo')
            return self
        self.meteo_data = MeteoFranceConnector(self.from_dt, self.to_dt).to_pandas()
        self.meteo_data = self.fix_types(self.meteo_data)
        if self.debug:
            log(f'{len(self.meteo_data)} lignes chargées sur meteo_data pour la période de {self.meteo_data.DATE.min()} à {self.meteo_data.DATE.max()} UTC')
        if self.cache or self.update_cache:
            self.meteo_data.to_hdf(self.meteo_store, key='meteo', mode='w', complevel=7)
            # self.meteo_data.to_csv(self.meteo_csv)
        return self

    def transform(self, params : Union[dict, None] = None):
        '''
        Transform the data according to params.
        '''
        if params:
            self.params = self.params | params
        if self.params['velib']:
            self.velib_data = transform_velib(self.velib_data, self.params, self.debug)
        if self.params['meteo']:
            self.meteo_data = transform_meteo(self.meteo_data, self.debug)
        if self.params['velib'] and self.params['meteo']:
            self.data = pd.merge(self.velib_data, self.meteo_data, on='datehour', how='left')
            log(f'After merge {self.data.isna().sum().sum()} NaN values to be dropped.')
            self.data = self.data.dropna()
        elif self.params['velib']:
            self.data = self.velib_data
        elif self.params['meteo']:
            self.data = self.meteo_data            
        self.data = self.fix_types(self.data) # type: ignore
        return self

    def fix_types(self, df : pd.DataFrame):
        '''
        Fix types to avoid fitting errors.
        '''
        proper_types = {
            'station' : str,
            'lat' : float,
            'lon' : float,
            'name' : str,
            'capacity' : float,
            'bikes' : float,
            'delta' : float
        }
        proper_types = {k:v for k,v in proper_types.items() if k in df.columns}
        return df.astype(proper_types) # type: ignore

##########################
###         ETL        ###
##########################

def transform_velib(velib_data, params : dict, debug):
    '''
    General procedure for velib data transformation.
    '''
    # TODO: Utiliser debug comme None | logger pour loguer les messages.
    base_columns = ['datehour', 'date', 'station', 'lat', 'lon', 'name']
    add_columns = []
    target_columns = ['capacity', 'bikes', 'delta']
    if debug:
        log('Transformation: velib_data')
    # Creation de DELTA
    velib_data['delta'] = velib_data.groupby('station')['bikes'].diff().fillna(velib_data.delta.astype(int))

    ## Suppression de doublons complets 
    if debug:
        full_duplicates_count = velib_data.duplicated(['dt', 'bikes', 'capacity', 'station']).sum()
        log(f"""Suppression de {full_duplicates_count} ({round(full_duplicates_count/len(velib_data)*100, 2)}%) doublons d'origine, c'est-à-dire des lignes identiques par les valeurs clé de données de temps réel Vélib: 'dt', 'bikes', 'capacity' et 'station'.""")            
    velib_data.drop_duplicates(['dt', 'bikes', 'capacity', 'station'], inplace=True)
    ## Correction de capacité 0
    velib_data['capacity'] = velib_data.groupby('station')['bikes'].transform('max')
    ## Introduire datehour
    velib_data['datehour'] = velib_data['dt'].dt.floor('h')
    ### Convert original UTC dt to Paris
    velib_data['datehour'] = velib_data['datehour']
    ## Merge (drop) les données multiple pour une heure
    if debug:
        calc_duplicated = velib_data.duplicated(['datehour', 'station']).sum()
        log(f"""Il y a {calc_duplicated} ({round(calc_duplicated/len(velib_data)*100, 2)}%) lignes qui réprésentent les mêmes stations pour les même dates et heures. On enlève les valeurs intrahoraires et recalcule les deltas.""")
    velib_data = velib_data.sort_values(['dt', 'status_id']).drop_duplicates(['datehour', 'station'])

    ## NA values
    if debug:
        log(f"""Suppression de {velib_data.isna().sum().sum()} valeurs manquantes.""")
    velib_data = velib_data.dropna()

    # Reconstruction de sésonalité pour avoir des propres deltas
    if debug: log('Reconstruction de saisonalité.')
    velib_data = velib_reconstruct_seasonality(velib_data)
    velib_data['date'] = velib_data.datehour.dt.date
    # Que faire avec les stations peu représentées
    if params['drop_stations_outliers']:
        blacklist_stations, norm_stations = velib_detect_outliers(velib_data.dropna().station.value_counts(), k=3, debug=debug)
        if debug:
            log(f'On enlève {len(blacklist_stations)} stations peu représentées dans le dataset.')
        len_before = len(velib_data)
        velib_data = velib_data[velib_data.station.isin(norm_stations)]
        len_after = len(velib_data)
        if debug:
            log(f'On a supprimé {len_before-len_after} lignes, {round((len_before-len_after) * 100/len_before, 2)}%.')

    # Ajouter les information de jour feries etc.
    velib_data = add_datetime_details(velib_data)
    add_columns.extend(['hour', 'month', 'weekday', 'weekend', 'holiday', 'preholiday', 'postholiday', 'pont', 'vacances', 'vacances_uni'])
    if params['reconstruct_velib']:
        if debug:
            log('Reconstruction de données: interpolation.')
        velib_data = velib_recontruct_missing_data_basic(velib_data, debug)
        add_columns.append('reconstructed')

    if params['drop_days_outliers']:
        # detect days outliers
        blacklist_days, norm_days = velib_detect_outliers(velib_data.dropna().date.value_counts(), k=3, debug=debug)
        if debug:
            log(f'On enlève {len(blacklist_days)} jours avec trop de données manquantes.')
        len_before = len(velib_data.dropna())
        velib_data = velib_data[velib_data.date.isin(norm_days)]
        len_after = len(velib_data.dropna())
        if debug:
            log(f'On a supprimé {len_before-len_after} ligne, {round((len_before-len_after) * 100/len_before, 2)}%.')

    if params['reconstruct_velib']:
        if debug:
            log('Reconstruction de données: mise en place de médianes saisonières.')
        velib_data = velib_recontruct_missing_data_extended(velib_data, debug)
        # add_columns.append('disabled')
        if debug:
            log(f"Reconstuit {velib_data['reconstructed'].sum()} enregistrements soit {round(velib_data['reconstructed'].sum() * 100 / len(velib_data.dropna()), 2)}%")

    # Choix final de colonnes à retourner.
    velib_data = velib_data[base_columns + add_columns + target_columns]
    return velib_data

def transform_meteo(meteo_data, debug):
    '''
    General transformation of MeteoFrance data.
    '''
    if debug:
        log('Transformation: meteo_data')
    rename_cols = {
        'T' : 'temp',
        'RR1' : 'precip',
        'DG' : 'gel',
        'FF' : 'vent'
    }
    meteo_data['datehour'] = pd.to_datetime(meteo_data.DATE, format='%Y%m%d%H')
    ### Convert original UTC dt to Paris and return to tz-unaware format
    meteo_data['datehour'] = meteo_data['datehour'].dt.tz_localize(tz='UTC').dt.tz_convert(tz='Europe/Paris').dt.tz_localize(None)

    meteo_data = meteo_data.dropna(axis=1, how='all')[['datehour'] + list(rename_cols.keys())].rename(rename_cols, axis=1)
    for col in meteo_data.select_dtypes('object').columns:
        meteo_data[col] = meteo_data[col].str.replace(',', '.').astype(float)
    seasonal = pd.DataFrame({'datehour' : pd.date_range(meteo_data.datehour.min(), meteo_data.datehour.max(), freq='h', tz='Europe/Paris').tz_convert(None)})
    # Ajoutons les données météo
    meteo_data = pd.merge(seasonal, meteo_data[['datehour', 'temp', 'precip', 'gel', 'vent']], on=['datehour'], how='left')
    meteo_data['precip'] = meteo_data['precip'].fillna(0) # Pas de précipitation si 0
    meteo_data['gel'] = meteo_data['gel'].fillna(0) # Pas de gel si 0
    meteo_data = meteo_data.interpolate() # Interpolate other
    return meteo_data


def velib_reconstruct_seasonality(velib_data : pd.DataFrame):
    '''
    Reconstruct date-hours sequence for velib data.
    '''
    stations = velib_data[['station', 'lat', 'lon', 'name', 'capacity']].drop_duplicates()
    datehours = pd.DataFrame({'datehour' : pd.date_range(velib_data.datehour.min(), velib_data.datehour.max(), freq='h', tz='Europe/Paris').tz_convert(None)})
    # Full outer join pour avoir la série reconstruite
    seasonal = datehours.merge(stations, how='cross')
    # Ajoutons les données vélos
    velib_data = pd.merge(seasonal, velib_data[['datehour', 'station', 'bikes']], on=['datehour', 'station'], how='left')
    # Construisons les deltas
    velib_data['delta'] = velib_data.groupby('station')['bikes'].diff()
    return velib_data

def velib_recontruct_missing_data_basic(velib_data : pd.DataFrame, debug):
    '''
    Basic reconstruction (linear interpolation) of data for short intervals (2 for any hours, 3 for the night).
    '''
    # Marquer les données reconsruite
    # MaJ: On ne considère pas l'interpolation linéaire de périodes courtes comme une reconstruction à noter 
    # velib_data['reconstructed'] = velib_data.bikes.isna().where(velib_data.bikes.notna(), np.nan)
    velib_data['bikes'] = velib_data.groupby('station').bikes.transform(lambda x: interpolate_sequence(x, 2))
    low_usage_mask = (velib_data.datehour.dt.hour < 6) | (velib_data.datehour.dt.hour > 22)
    velib_data.loc[low_usage_mask, 'bikes'] = velib_data[low_usage_mask].groupby(['station']).bikes.transform(lambda x: interpolate_sequence(x, 3))
    # reconstruction de delta
    velib_data['delta'] = velib_data.groupby('station')['bikes'].diff()
    # Marquer les données reconsruite
    # MaJ: On ne considère pas l'interpolation linéaire de périodes courtes comme une reconstruction à noter 
    # velib_data['reconstructed'] = velib_data['reconstructed'].fillna(velib_data.bikes.notna().where(velib_data.bikes.notna(), np.nan))
    return velib_data

def velib_recontruct_missing_data_extended(velib_data : pd.DataFrame, debug):
    # Marquer les données reconsruite
    # reconstructed = no delta existed before extended reconstruction
    velib_data['reconstructed'] = velib_data.delta.isna() #.where(velib_data.delta.notna(), np.nan)
    # Interpolation à la base de médiane saisonière pour les données manquantes
    velib_data['delta'] = velib_data.groupby('station', group_keys=False)[['weekday', 'hour', 'delta']].apply(interpolate_seasonality_trend)
    if debug:
        log('Recalculating bikes numbers.')
    velib_data['bikes'] = velib_data.groupby('station', group_keys=False)[['delta', 'bikes']].apply(fill_bikes)
    # On ne peut pas avoir bikes négatifs
    velib_data['bikes'] = velib_data['bikes'].where(velib_data['bikes'] >= 0, 0)
    ## Recalcule de delta
    velib_data['delta'] = velib_data.groupby('station')['bikes'].diff().fillna(velib_data.delta.astype(int))
    log(f"After reconstruction {velib_data['delta'].isna().sum()} NA deltas.")
    # Marquer les données reconsruite
    # reconstructed = true if reconstructed was NA and delta has been set, else reconstructed is NA (we were unable to reconstruct the data)
    # velib_data['reconstructed'] = velib_data['reconstructed'].fillna(True).astype(bool)
    return velib_data


def fill_bikes(group : pd.DataFrame):
    '''
    Restore bikes' numbers from delta values.
    '''
    if group.bikes.isna().all() or not group.bikes.isna().any():
        return group.bikes
    group = group.copy()
    bikes = group['bikes'].to_list()
    deltas = group['delta'].to_list()
    first_not_na = 0
    for first_not_na in range(len(bikes)):
        if pd.notna(bikes[first_not_na]):
            break
    if first_not_na > 0:
        # Descente en reconstruction
        for i in range(first_not_na - 1, -1, -1):
            bikes[i] = bikes[i + 1] - deltas[i + 1]
    if first_not_na < len(group) - 1:
        # Remonté en reconstruction
        for i in range(first_not_na + 1, len(bikes)):
            bikes[i] = bikes[i - 1] + deltas[i]
    return group['bikes'].fillna(pd.Series(bikes, index=group.index))

def velib_detect_outliers(selection: pd.Series, k = 1.5, method='IQR', debug = True):
    """
    Get a blacklist of indexes considered as outliers.
    Args:
        - method = 'IQR' (default) ou 'zscore'
        - k - multiply coef for IQR or zscore limit
    """
    if method == 'IQR':
        q = selection.quantile([0.25, 0.75]).to_list()
        IQR = (q[1] - q[0]) * k
        low_border = q[0] - IQR
        high_border = q[1] + IQR
    elif method=='zscale':
        selection_zscore = selection.copy()
        selection_zscore = zscore(selection)
        low_border = selection[selection_zscore < -k].max()
        high_border = selection[selection_zscore > k].min()
    elif method=='hard':
        low_border = k
        high_border = selection.max()
    else:
        raise SyntaxError('Wrong method.')
    high_outliers = selection[selection>high_border].index.to_list()
    low_outliers = selection[selection<low_border].index.to_list()
    norm_stations = selection[(selection>=low_border)|(selection<=high_border)].index.to_list()
    if debug:
        log('Valeur min-max:', selection.min(), '-', selection.max())
        log('Seuils de outliers:', low_border, '-', high_border)
        log('Nombre de valeurs total:', len(selection))
        log(f'Grands outliers: {len(high_outliers)} ou {round(len(high_outliers)/len(selection)*100, 2)}%')
        log(f'Petits outliers: {len(low_outliers)} ou {round(len(low_outliers)/len(selection)*100, 2)}%')
    return high_outliers + low_outliers, norm_stations


### Helpers ###



def load_holidays():
    """Chargement des dates de jours fériés en France métropolitaine
    https://www.data.gouv.fr/fr/datasets/jours-feries-en-france-metropolitaine/

    Returns:
        pandas.Series[str] : '%Y-%m-%d'

    Depends on:
    data/om-referentiel-jours-feries.csv
    
    """
    df = pd.read_csv(Storage.shared_data('om-referentiel-jours-feries.csv'), sep=';')
    # df = pd.read_csv(r'data/om-referentiel-jours-feries.csv', sep=';')
    return df.date

def load_vacances_scolaires():
    '''
    Add school holidays.
    '''
    # df_vacances = pd.read_csv(r'data/fr-en-calendrier-scolaire.csv', sep=';', parse_dates=['Date de début', 'Date de fin'])
    df_vacances = pd.read_csv(Storage.shared_data('fr-en-calendrier-scolaire.csv'), sep=';', parse_dates=['Date de début', 'Date de fin'])
    df_vacances = df_vacances[(df_vacances['Académies'] == 'Paris') & (df_vacances['Date de début'].dt.year >= 2024)]
    return df_vacances

def load_vacances_sorbonne():
    '''
    HARDCODED VALUES! Add university holidays based on Sorbonne planning.
    '''
    starts = [
        datetime.date(2024, 10, 27),
        datetime.date(2024, 12, 23),
        datetime.date(2025, 2, 24),
        datetime.date(2025, 5, 14),
        ]
    ends = [
        datetime.date(2024, 11, 3),
        datetime.date(2025, 1, 5),
        datetime.date(2025, 3, 2),
        datetime.date(2025, 5, 21),
    ]
    return pd.DataFrame({'start' : starts, 'end' : ends})

def add_datetime_details(df : pd.DataFrame, use_holidays = True, datehour : str = 'datehour'):
    """Ajout de colonnes:
        'hour' : 0..23, 
        'month' : 1..12, 
        'weekday': 0..6, 
        'weekend' : 0..1,
        'holiday' : 0..1,
        'preholiday' : 0..1
        'postholiday' : 0..1
        'pont' : 0..1 (vendredi postholiday)
    Args:
        df (pd.DataFrame): dataset avec la colonne 'datehour'
        use_holidays (bool, optional): S'il faut charger les jours fériés et ajouter la col 'holiday'. Defaults to True.

    Returns:
        df mis à jour
    """
    df['hour'] = df[datehour].dt.hour
    df['month'] = df[datehour].dt.month
    df['date'] = df[datehour].dt.date
    df['weekday'] = df[datehour].dt.weekday
    df['weekend'] = df.weekday.isin([5, 6]).astype(int)
    if use_holidays:
        holidays = load_holidays()
        df['holiday'] = (df[datehour].dt.strftime(r'%Y-%m-%d').isin(holidays)).astype(int)
        df['preholiday'] = ((df[datehour] + pd.DateOffset(days=1)).dt.strftime(r'%Y-%m-%d').isin(holidays) & df['holiday'].eq(0)).astype(int)
        df['postholiday'] = ((df[datehour] + pd.DateOffset(days=-1)).dt.strftime(r'%Y-%m-%d').isin(holidays) & df['holiday'].eq(0)).astype(int)
        df['pont'] = (df['postholiday'].eq(1) & df['weekday'].eq(4)).astype(int) #vendredi postholiday faison pont!!!
        vacances = load_vacances_scolaires()
        df['vacances'] = 0
        for i, row in vacances.iterrows():
            df['vacances'] = df['vacances'].where(~df['date'].between(row['Date de début'].date(), row['Date de fin'].date()), 1)
        vacances_uni = load_vacances_sorbonne()
        df['vacances_uni'] = 0
        for i, row in vacances_uni.iterrows():
            df['vacances_uni'] = df['vacances_uni'].where(~df['date'].between(row['start'], row['end']), 1)
    return df

def interpolate_sequence(data : pd.Series, max_len=3):
    """Faire l'intérpolation linéaire pour les séquences de na qui ne sont pas plus longues que max_len
    """
    label = data.name
    idata : pd.DataFrame = data.copy().to_frame()
    # Labélisé les groupes de na/non na
    idata['_group'] = idata[label].isna().astype(int).diff().ne(0).cumsum()
    groups_sizes = idata[idata[label].isna()].groupby('_group').size()
    if max_len != -1:
        selected_groups = groups_sizes[groups_sizes <= max_len]
    else:
        selected_groups = groups_sizes
    mask = idata['_group'].isin(selected_groups.index)
    idata.loc[mask, label] = idata[label].interpolate(method='linear') # type: ignore
    return idata[label]

def interpolate_seasonality_trend(group : pd.DataFrame):
    '''
    Restore missing deltas by interpolating seasonal medians (by weekday and hour) calculated on 3 neighbouring values.
    '''
    if not group.delta.isna().any():
        return group['delta']
    group = group.copy()
    # calcule de mediane saisonière pour le jour et l'heure
    group['seasonal_median'] = group.groupby(['weekday', 'hour'], group_keys=False)['delta'].apply(lambda g: g.rolling(3, min_periods=1).median())
    # ajout de valeurs voisines
    group['last_seasonality'] = group['seasonal_median'].ffill().bfill()
    group['next_seasonality'] = group['seasonal_median'].bfill().ffill()
    # seasonal median serait le moyen de données saisonières
    group['seasonal_median'] = group['seasonal_median'].fillna(group[['last_seasonality', 'next_seasonality']].mean(axis=1, skipna=True))
    # group['seasonal_median'] = group.groupby(['weekday', 'hour'], group_keys=False)['seasonal_median'].apply(lambda g: g.ffill().bfill())
    assert group['seasonal_median'].notna().all(), f"Seasonal median has {group['seasonal_median'].isna().sum()} NaNs for {len(group)} values with a total of {group['delta'].isna().sum()} NaN delta."
    # Remplacer les données manquantes pas les médianes saisonières lissées
    group['delta'] = group['delta'].fillna(group['seasonal_median'])
    return group['delta']

def na_groups(group : pd.DataFrame, compare_to : str = 'category', source : str = 'value', target : str = 'wow', reverse=False):
    '''
    Mark NA sequences with unique number labels.
    '''
    notna_holder = "42" if group[compare_to].dtype.kind in 'biufc' else 1
    group = group.copy()
    groups = (group[compare_to].isna() != group[compare_to].shift(1, fill_value=notna_holder).isna()).astype(int).cumsum() * group[compare_to].isna()
    # group['g'] = groups
    if reverse:
        group[target] = group[groups!=0].groupby(groups, group_keys=False)[source].apply(lambda x: x[::-1].cumsum()[::-1])        
    else:
        group[target] = group[groups!=0].groupby(groups)[source].cumsum()
    return group[target]


def detect_outliers(selection: pd.Series, k = 1.5, method='IQR'):
    """
    Returns a blacklist of indexes considered as outliers.
    Args:
        - method = 'IQR' (default) ou 'zscore'
        - k - multiply coef for IQR or zscore limit
    """
    if method == 'IQR':
        q = selection.quantile([0.25, 0.75]).to_list()
        IQR = (q[1] - q[0]) * k
        low_border = q[0] - IQR
        high_border = q[1] + IQR
    elif method=='zscale':
        selection_zscore = selection.copy()
        selection_zscore = zscore(selection)
        low_border = selection[selection_zscore < -k].max()
        high_border = selection[selection_zscore > k].min()
    elif method=='hard':
        low_border = k
        high_border = selection.max()
    else:
        raise SyntaxError('Wrong method.')
    log('Valeur min-max:', selection.min(), '-', selection.max())
    log('Seuils de outliers:', low_border, '-', high_border)
    high_outliers = selection[selection>high_border].index.to_list()
    low_outliers = selection[selection<low_border].index.to_list()
    log('Nombre de valeurs total:', len(selection))
    log(f'Grands outliers: {len(high_outliers)} ou {round(len(high_outliers)/len(selection)*100, 2)}%')
    log(f'Petits outliers: {len(low_outliers)} ou {round(len(low_outliers)/len(selection)*100, 2)}%')
    return high_outliers + low_outliers

def convert_lon(lon, min_lon):
    """
    Get meters distance for lon degrees.
    """
    lon -= min_lon
    return lon * 72987
def convert_lat(lat, min_lat):
    """
    Get meters distance for lat degrees.
    """
    lat -= min_lat
    return lat * 111000

### Monitoring

def save_dataset_stats(df : pd.DataFrame, values : list[str] = ['delta', 'temp', 'precip', 'vent'], ignore_monthly=False):
    '''
    Save statistical information for EXTRACTED_DATASET to local/data/stats/YYYYmmddHH-YYYYmmddHH.json
    '''
    # df structure: dt, poll_dt
    min_datehour = df['datehour'].min()
    max_datehour = df['datehour'].max()
    last_month = max_datehour - datetime.timedelta(days=30)
    previous_month = max_datehour - datetime.timedelta(days=60)
    log('Calculating dataset stats...')
    filename = Storage.stats(f"{max_datehour.strftime('%Y%m%d%H')}-from-{min_datehour.strftime('%Y%m%d%H')}.json")
    stats = {
        'full' : get_dataset_stats(df, values),
    }
    if not ignore_monthly:
        stats['30d'] = get_dataset_stats(df[df['datehour'] >= last_month])
        stats['30d']['kstest'] = get_kstest_for_values(df[df['datehour'] >= last_month], df[(df['datehour'] < last_month) & (df['datehour'] >= previous_month)])
    log(f'Saving dataset stats to {filename}.')
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=4)

def get_dataset_stats(df : pd.DataFrame, values : list[str] = ['delta', 'temp', 'precip', 'vent']) -> dict:
    '''
    Get statistical information for EXTRACTED_DATASET or it's part.
    '''
    stats = {
        'count' : {
            'all' : len(df),
            'na' : len(df[df.isna().any(axis=1)])
        },
        'stats' : {}
    }    
    for val in values:
        q = df[val].quantile([0.25, 0.5, 0.75]).to_list()
        IQR = (q[2] - q[0]) * 1.5
        stats['count'][val + '_na'] = len(df[df[val].isna()])
        stats['count'][val + '_outliers_top'] = len(df[df[val] > q[2] + IQR])
        stats['count'][val + '_outliers_bottom'] = len(df[df[val] < q[0] - IQR])
        stats['count'][val + '_outliers'] = stats['count'][val + '_outliers_top'] + stats['count'][val + '_outliers_bottom']
        stats['stats'][val + '_q1'] = q[0]
        stats['stats'][val + '_q2'] = q[1]
        stats['stats'][val + '_q3'] = q[2]
        stats['stats'][val + '_high'] = q[2] + IQR
        stats['stats'][val + '_low'] = q[0] + IQR
        stats['stats'][val + '_min'] = df[val].min()
        stats['stats'][val + '_max'] = df[val].max()
    return stats

def get_kstest_for_values(df_a: pd.DataFrame, df_b: pd.DataFrame, values : list[str] = ['delta', 'temp', 'precip', 'vent']):
    '''
    Get Kholmogorov-Smirnov test scores for values for samples
    '''
    stats = {
        val : kstest(df_a[val], df_b[val]).statistic
        for val in values
    }
    return stats
