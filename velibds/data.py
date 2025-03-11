import datetime, sys
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from velibds.connect import VelibConnector, MeteoFranceConnector
from velibdslib import *
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from pathlib import Path


class VelibData:
    params = {
        'clusters' : True,
        'drop_outliers' : True,
        'reconstruct_velib' : True,
        'velib' : True,
        'meteo' : True,
    }

    velib_csv = r'local_data/velib_orig.csv'
    meteo_csv = r'local_data/meteo_orig.csv'
    impute = False

    def __init__(self, from_dt : datetime.datetime = datetime.date(2024, 12, 5), 
                 to_dt : datetime.datetime = None, debug = True, 
                 params : dict = None, cache = False, update_cache = False):
        if to_dt is None:
            to_dt = datetime.date.today()
        if params:
            self.params = self.params | params
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.debug = debug
        self.cache = cache
        self.update_cache = update_cache

    def extract(self):
        if self.params['velib']:
            self.extract_velib()
        if self.params['meteo']:
            self.extract_meteo()
        return self

    def extract_velib(self):
        if self.cache and not self.update_cache and Path(self.velib_csv).is_file():
            self.velib_data = pd.read_csv(self.velib_csv, parse_dates=['dt'])
            return self
        cmd = f"""
        SELECT status_id, station, lat, lon,
            delta, bikes, capacity, name,
                dt from velib_all
                where dt > '{self.from_dt.strftime(r'%Y-%m-%d')}' {"and dt <'" + self.to_dt.strftime(r'%Y-%m-%d') + "'" if self.to_dt else ""}
        """
        self.velib_data = VelibConnector(cmd).to_pandas()
        if self.debug:
            print(f'{len(self.velib_data)} lignes chargées sur velib_data pour la période de {self.velib_data.dt.min().date()} à {self.velib_data.dt.max().date()}')     
        if self.cache or self.update_cache:
            self.velib_data.to_csv(self.velib_csv)
        return self

    def extract_meteo(self):
        if self.cache and not self.update_cache and Path(self.meteo_csv).is_file():
            self.meteo_data = pd.read_csv(self.meteo_csv)
            return self
        self.meteo_data = MeteoFranceConnector(self.from_dt, self.to_dt).to_pandas()
        if self.debug:
            print(f'{len(self.meteo_data)} lignes chargées sur meteo_data pour la période de {self.meteo_data.DATE.min()} à {self.meteo_data.DATE.max()}')
        if self.cache or self.update_cache:
            self.meteo_data.to_csv(self.meteo_csv)
        return self
        
    def transform(self, params : dict = None):
        if params:
            self.params = self.params | params
        if self.params['velib']:
            self.transform_velib()
        if self.params['meteo']:
            self.transform_meteo()
        if self.params['velib'] and self.params['meteo']:
            self.data = pd.merge(self.velib_data, self.meteo_data, on='datehour', how='left')
        elif self.params['velib']:
            self.data = self.velib_data
        elif self.params['meteo']:
            self.data = self.meteo_data            
        if self.impute:
            ##Reconstruct missing data
            self.impute_data()
        self.fix_types()
        return self
    
    def fix_types(self):
        proper_types = {
            'station' : str,
            'lat' : float,
            'lon' : float,
            'name' : str,
            'cluster' : int,
            'capacity' : int,
            'bikes' : int,
            'delta' : int
        }
        proper_types = {k:v for k,v in proper_types.items() if k in self.data.columns}
        self.data = self.data.astype(proper_types)

    def transform_velib(self, params : dict = None):
        base_columns = ['datehour', 'station', 'lat', 'lon', 'name']
        add_columns = []
        target_columns = ['capacity', 'bikes', 'delta']
        if params:
            self.params = self.params | params
        if self.debug:
            print('Transformation: velib_data')
        ## NA values
        if self.debug:
            print(f"""Suppression de {self.velib_data.isna().sum().sum()} valeurs manquantes.""")
        self.velib_data.dropna()
        ## Suppression de doublons complets 
        if self.debug:
            full_duplicates_count = self.velib_data.duplicated(['dt', 'bikes', 'capacity', 'station']).sum()
            print(f"""Suppression de {full_duplicates_count} ({round(full_duplicates_count/len(self.velib_data)*100, 2)}%) doublons d'origine, c'est-à-dire des lignes identiques par les valeurs clé de données de temps réel Vélib: 'dt', 'bikes', 'capacity' et 'station'.""")            
        self.velib_data.drop_duplicates(['dt', 'bikes', 'capacity', 'station'], inplace=True)
        ## Correction de capacité 0
        def fix_capacity(group):
            group['capacity'] = group.bikes.max()
            return group['capacity']
        self.velib_data['capacity'] = self.velib_data.groupby('station', group_keys=False)[['bikes','capacity']].apply(fix_capacity)
        ## Introduire datehour
        self.velib_data['datehour'] = self.velib_data.dt.dt.floor('h')
        ## Merge (drop) les données multiple pour une heure
        if self.debug:
            calc_duplicated = self.velib_data.duplicated(['datehour', 'station']).sum()
            print(f"""Il y a {calc_duplicated} ({round(calc_duplicated/len(self.velib_data)*100, 2)}%) lignes qui réprésentent les mêmes stations pour les même dates et heures. On enlève les valeurs intrahoraires et recalcule les deltas.""")
        self.velib_data = self.velib_data.sort_values(['dt', 'status_id']).drop_duplicates(['datehour', 'station'])
        if self.params['drop_outliers']:
            blacklist_stations = [] + detect_outliers(self.velib_data.station.value_counts(), k=3)
            if self.debug:
                print(f'On enlève {len(blacklist_stations)} stations peu représentées dans le dataset.')
            len_before = len(self.velib_data)
            self.velib_data = self.velib_data[~self.velib_data.station.isin(blacklist_stations)]
            len_after = len(self.velib_data)
            if self.debug:
                print(f'On a supprimé {len_before-len_after} ligne, {round((len_before-len_after)/len_before, 2)}%.')
            
        if self.params['clusters']:
            ## Ajoute des clusters
            self.add_clusters()
            add_columns.append('cluster')
        if self.params['reconstruct_velib']:
            self.recontruct_seasonality()
            add_columns.append('reconstructed')
            add_columns.append('disabled')
        ## Columns to keep: datehour, station, lat, lon, name, cluster, bikes
        # self.velib_data = self.velib_data[['datehour', 'station', 'lat', 'lon', 'name', 'cluster', 'capacity', 'bikes', 'delta']]
        self.velib_data = self.velib_data[base_columns + add_columns + target_columns]
        self.velib_data = add_datetime_details(self.velib_data)
        return self
    
    def add_clusters(self):
        self.velib_data = pd.merge(self.velib_data, self.get_clusters_velib()[['station', 'cluster', 'convlat', 'convlon']], on='station', how='left')
        if self.params['drop_outliers']:
            ## Il y a des clusters trop peu représentées
            datehour_df = self.velib_data.groupby('cluster').datehour.nunique()
            q = datehour_df.quantile([0.25, 0.75]).to_list()
            # outliers comme q1 - delta(q3,q1)*3 et q3 + delta(q3,q1)*3
            q_margin = (q[1] - q[0]) * 3 if q[1] != q[0] else 10 # arbitrairement on prend la différence de 10 heures comme importante si les données sont bien concentrées 
            seuil_bas = q[0] - q_margin
            # seuil_haut = q[1] + q_margin
            bottom_outliers = datehour_df[datehour_df<seuil_bas].index
            bottom_outliers_count = len(bottom_outliers)
            orig_count = len(self.velib_data)
            self.velib_data = self.velib_data[~self.velib_data.cluster.isin(bottom_outliers)]
            cleaned_count = len(self.velib_data)
            if self.debug:
                print('Suppression des clusters peu représentés')
                print('Max heures par clusters:', datehour_df.max())
                print('Min heures par clusters:', datehour_df.min())
                print('Seuil bas de clusters outliers:', seuil_bas)
                print("Total de clusters:", self.velib_data.cluster.nunique())
                print("Nombre de clusters à supprimer:", bottom_outliers_count)
                print(f"Perte de données: {orig_count - cleaned_count} lignes, {round(100*(orig_count - cleaned_count)/orig_count, 2)}%")

    def get_clusters_velib(self):
        stations = self.velib_data[['station', 'lat', 'lon', 'name']].drop_duplicates()
        stations['lat'] = stations.lat.apply(float)
        stations['lon'] = stations.lon.apply(float)
        min_lon = stations.lon.min()
        min_lat = stations.lat.min()
        stations['convlon'] = stations.lon.apply(convert_lon, args=[min_lon])
        stations['convlat'] = stations.lat.apply(convert_lat, args=[min_lat])
        n_clusters = get_best_silhouette_score(stations)
        if self.debug:
            print(f'Calcule de {n_clusters} clusters')
        model = KMeans(n_clusters=n_clusters, random_state=0) if self.debug else KMeans(n_clusters=n_clusters)
        kmeans = model.fit(stations[['convlat', 'convlon']])
        stations['cluster'] = kmeans.labels_
        return stations

    def recontruct_seasonality(self):
        # stations = self.velib_data[['station', 'lat', 'lon', 'convlat', 'convlon', 'name', 'cluster', 'capacity']].drop_duplicates()
        stations = self.velib_data[['station', 'lat', 'lon', 'name', 'cluster', 'capacity']].drop_duplicates()
        datehours = pd.DataFrame({'datehour' : pd.date_range(self.velib_data.datehour.min(), self.velib_data.datehour.max(), freq='h')})
        seasonal = datehours.merge(stations, how='cross')
        self.velib_data = pd.merge(seasonal, self.velib_data[['datehour', 'station', 'bikes', 'delta']], on=['datehour', 'station'], how='left')
        # Recalcule de delta pour la série continue (sinon certains deltas on monté la différence entre les heures bien décalées)
        self.velib_data['delta'] = self.velib_data.groupby('station')['bikes'].diff()
        self.velib_data['reconstructed'] = self.velib_data.delta.isna()
        self.velib_data['delta'] = self.velib_data.groupby('station', group_keys=False)['delta'].apply(self.fill_long_sequence, fill_with=np.inf)
        self.velib_data['disabled'] = np.isinf(self.velib_data.delta)
        self.velib_data['delta'] = self.velib_data['delta'].replace(to_replace=np.inf, value=0) 
        na_count = self.velib_data.bikes.isna().sum()
        if self.debug:
            print(f'Pour la série temporelle continue il y a {na_count} lignes à reconstruire des valeurs manquantes, soit {round(100*na_count/len(self.velib_data), 2)}%')
        if na_count:
            self.impute = True

    def impute_data(self):
        if 'delta' not in self.data.columns:
            print('Unable to impute data: no delta column.')
            return
        if self.debug:
            print('Running imputer.')
        cols = ['delta', 'weekday', 'hour', 'weekend', 'holiday', 'preholiday']
        if self.params['meteo']:
            cols += ['temp', 'precip', 'gel', 'vent']
        # self.data['delta'] = self.data.groupby('station', group_keys=False)[cols].apply(self.impute_regressor_stations)
        # # # Si tjrs des deltas vides:
        # if self.data['delta'].isna().sum():
        #     self.data['delta'] = self.impute_regressor_all(self.data)
        if self.debug:
            print('interpolate_seasonality_trend')
        self.data['delta'] = self.data.groupby('station', group_keys=False)[['weekday', 'hour', 'delta']].apply(self.interpolate_seasonality_trend)
        if self.debug:
            print('Recalculating bikes numbers.')
        self.data['bikes'] = self.data.groupby('station', group_keys=False)[['delta', 'bikes']].apply(self.fill_bikes)
        # On ne peut pas avoir bikes négatifs
        self.data['bikes'] = self.data['bikes'].where(self.data['bikes'] >= 0, 0)
        ## Recalcule de delta
        self.data['delta'] = self.data.groupby('station')['bikes'].diff().fillna(self.data.delta.astype(int))

    def fill_bikes(self, group):
        group = group.copy()
        group['last_bikes'] = group.bikes.ffill()
        group['add_delta'] = na_groups(group, compare_to='bikes', source='delta')
        group['bikes'] = group['bikes'].fillna(group['last_bikes'] + group['add_delta'])
        group['next_bikes'] = group.bikes.bfill()
        group['next_delta'] = group.delta.shift(-1)
        group['sub_delta'] = na_groups(group, compare_to='bikes', source='next_delta', reverse=True)
        return group['bikes'].fillna(group['next_bikes'] - group['sub_delta'])
    
    # def impute_velib(self):
    #     cols = ['delta', 'reconstructed', 'weekday', 'hour', 'weekend', 'holiday', 'temp', 'precip', 'gel', 'vent', 'capacity']
    #     self.velib_data['delta'] = self.velib_data.groupby('station', group_keys=False)[cols].apply(self.impute_regressor_stations)
    #     # Si tjrs des deltas vides:
    #     if self.velib_data['delta'].isna().sum():
    #         self.velib_data['delta'] = self.impute_regressor_all(self.velib_data)
    
    def fill_long_sequence(self, data : pd.Series, max_len=24, fill_with=0):
        """Si on n'a pas de mise à jours depuis une journée - considère la station inactive pour la période
        """
        label = data.name
        data = data.copy().to_frame()
        # Labélisé les groupes de na/non na
        data['_group'] = data[label].isna().astype(int).diff().ne(0).cumsum()
        groups_sizes = data[data[label].isna()].groupby('_group').size()
        selected_groups = groups_sizes[groups_sizes > max_len]
        mask = data['_group'].isin(selected_groups.index)
        data.loc[mask, label] = fill_with
        return data[label]
        
    def interpolate_sequence(self, data : pd.Series, max_len=3):
        """Faire l'intérpolation linéaire pour les séquences de na qui ne sont pas plus longues que max_len
        """
        label = data.name
        data = data.copy().to_frame()
        # Labélisé les groupes de na/non na
        data['_group'] = data[label].isna().astype(int).diff().ne(0).cumsum()
        groups_sizes = data[data[label].isna()].groupby('_group').size()
        if max_len != -1:
            selected_groups = groups_sizes[groups_sizes <= max_len]
        else:
            selected_groups = groups_sizes
        mask = data['_group'].isin(selected_groups.index)
        data.loc[mask, label] = data[label].interpolate(method='linear')
        return data[label]
    
    def interpolate_seasonality_trend(self, group : pd.DataFrame):
        if not group.delta.isna().any():
            return group['delta']
        group = group.copy()
        # calcule de mediane saisonière pour le jour et l'heure pour dernier 30 jours
        # delay = 30
        # group['seasonal_median'] = group.iloc[-delay:].groupby(['weekday', 'hour'], group_keys=False)[['delta']].apply(lambda g: g.delta.median())
        # group['seasonal_median'] = group.groupby(['weekday', 'hour']).rolling(delay, min_periods=1, center=True).delta.median(skipna=True).droplevel([0,1]).fillna(0)
        group['seasonal_median'] = group.groupby(['weekday', 'hour']).delta.transform('median')
        group['last_seasonality'] = group['seasonal_median'].ffill()
        group['next_seasonality'] = group['seasonal_median'].bfill()
        group['seasonal_median'] = group['seasonal_median'].fillna(group[['last_seasonality', 'next_seasonality']].mean(axis=1, skipna=True))
        group['trend'] = group['delta'] - group['seasonal_median']
        # group['trend_mean'] = group['trend'].rolling(5, center=True).mean()
        group['last_trend'] = group['trend'].ffill()
        group['next_trend'] = group['trend'].bfill()
        group['trend'] = group['trend'].fillna(group[['last_trend', 'next_trend']].mean(axis=1, skipna=True))
        # group['trend_mean'] = group.trend.rolling(5, min_periods=1, win_type="gaussian", center=True).mean(std=1.0)
        # group['trend_mean'] = group.trend.rolling(5, min_periods=1, center=True).mean()
        assert group['seasonal_median'].notna().all(), f"Seasonal median has {group['seasonal_median'].isna().sum()} nas: {group['seasonal_median']}"
        assert group['trend'].notna().all(), f"Trend has {group['trend'].isna().sum()} nas: {group['trend']}"
        group['delta'] = group['delta'].fillna(group['seasonal_median'] + group['trend'])
        return group['delta']
    
    def impute_regressor_stations(self, group):
        # cols = ['weekday', 'hour', 'weekend', 'holiday', 'temp', 'precip', 'gel', 'vent']
        if group.delta.isna().sum() == 0 or group.delta.isna().sum() > group.delta.notna().sum()/3:
            return group['delta']
        X_train = group[group.delta.notna()].drop('delta', axis=1)
        y_train = group[group.delta.notna()]['delta']
        X_pred = group[group.delta.isna()].drop('delta', axis=1)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_pred = scaler.transform(X_pred)
        model = BayesianRidge()
        model.fit(X_train, y_train)
        mask = group.delta.isna() == True
        group.loc[mask, 'delta'] = model.predict(X_pred).astype(int)
        return group['delta']

    def impute_regressor_all(self, group):
        cols = ['lat', 'lon', 'weekday', 'hour', 'cluster', 'capacity', 'weekend', 'holiday', 'preholiday', 'temp', 'precip', 'gel', 'vent']
        # cols = ['weekday', 'hour', 'weekend', 'holiday', 'temp', 'precip', 'gel', 'vent']
        X_train = group[group.delta.notna()][cols]
        y_train = group[group.delta.notna()]['delta']
        X_pred = group[group.delta.isna()][cols]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_pred = scaler.transform(X_pred)
        model = BayesianRidge()
        model.fit(X_train, y_train)
        # group['delta_orig'] = group.delta
        mask = group.delta.isna() == True
        group.loc[mask, 'delta'] = model.predict(X_pred).astype(int)
        return group['delta']

    def transform_meteo(self):
        if self.debug:
            print('Transformation: meteo_data')
        rename_cols = {
            'T' : 'temp',
            'RR1' : 'precip',
            'DG' : 'gel',
            'FF' : 'vent'
        }
        self.meteo_data['datehour'] = pd.to_datetime(self.meteo_data.DATE, format='%Y%m%d%H')
        self.meteo_data = self.meteo_data.dropna(axis=1, how='all')[['datehour'] + list(rename_cols.keys())].rename(rename_cols, axis=1)
        for col in self.meteo_data.select_dtypes('object').columns:
            self.meteo_data[col] = self.meteo_data[col].str.replace(',', '.').astype(float)
        return self

    @staticmethod
    def help():
        from IPython.display import Markdown, display
        import inspect
        def printmd(dat: str): display(Markdown(dat))
        printmd('## VelibData functions and arguments')
        printmd('### VelibData():')
        display(dict(inspect.signature(VelibData).parameters))
        printmd('### extract():')
        display(dict(inspect.signature(VelibData().extract).parameters))
        printmd('### transform():')
        display(dict(inspect.signature(VelibData().transform).parameters))
        printmd('### params dict:')
        display(VelibData.params)
        printmd('## Examples')
        printmd('### Extract, transform and create united dataset for the period from 2024-12-05 to today:')
        printmd("""```df = VelibData().extract().transform().data```""")
        printmd("### Choose only February:")
        printmd("""```df = VelibData(from_dt = datetime.date(2025, 2, 1), to_dt = datetime.date(2025, 3, 1)).extract().transform().data```""")
        printmd("### Create united dataset but don't restore continuous time series and don't impute missing data:")
        printmd("""```df = VelibData(params={'reconstruct_velib' = False}).extract().transform().data```""")
        printmd("### Create united dataset without dropping stations/clusters outliers:")
        printmd("""```df = VelibData(params={'drop_outliers' = False}).extract().transform().data```""")
        printmd("### No clusters:")
        printmd("""```df = VelibData(params={'clusters' = False}).extract().transform().data```""")
        printmd("""### Don't add meteo data:""")
        printmd("""```df = VelibData(params={'meteo': False}).extract().transform().data```""")
        printmd("""### Use cache in local_data folder to save and load data (minimize online transactions):""")
        printmd("""```df = VelibData(cache=True).extract().transform().data```""")
        printmd("""### Recreate cache files with up to date data:""")
        printmd("""```df = VelibData(update_cache=True).extract().transform().data```""")


def load_holidays():
    """Chargement des dates de jours fériés en France métropolitaine
    https://www.data.gouv.fr/fr/datasets/jours-feries-en-france-metropolitaine/

    Returns:
        pandas.Series[str] : '%Y-%m-%d'

    Depends on:
    data/om-referentiel-jours-feries.csv
    
    """
    df = pd.read_csv(r'data/om-referentiel-jours-feries.csv', sep=';')
    return df.date

def add_datetime_details(df : pd.DataFrame, use_holidays = True, datehour : str = 'datehour'):
    """Ajout de colonnes:
        'hour' : 0..23, 
        'weekday': 0..6, 
        'weekend' : 0..1,
        'holiday' : 0..1,
        'preholiday' : 0..1
    Args:
        df (pd.DataFrame): dataset avec la colonne 'datehour'
        use_holidays (bool, optional): S'il faut charger les jours fériés et ajouter la col 'holiday'. Defaults to True.

    Returns:
        df mis à jour
    """
    df['hour'] = df[datehour].dt.hour
    df['weekday'] = df[datehour].dt.weekday
    df['weekend'] = df.weekday.isin([5, 6]).astype(int)
    if use_holidays:
        holidays = load_holidays()
        df['holiday'] = (df[datehour].dt.strftime(r'%Y-%m-%d').isin(holidays)).astype(int)
        df['preholiday'] = ((df[datehour] + pd.DateOffset(days=1)).dt.strftime(r'%Y-%m-%d').isin(holidays)).astype(int)
    return df

def na_groups(group : pd.DataFrame, compare_to : str = 'category', source : str = 'value', target : str = 'wow', reverse=False):
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
    print('Valeur min-max:', selection.min(), '-', selection.max())
    print('Seuils de outliers:', low_border, '-', high_border)
    high_outliers = selection[selection>high_border].index.to_list()
    low_outliers = selection[selection<low_border].index.to_list()
    print('Nombre de valeurs total:', len(selection))
    print(f'Grands outliers: {len(high_outliers)} ou {round(len(high_outliers)/len(selection)*100, 2)}%')
    print(f'Petits outliers: {len(low_outliers)} ou {round(len(low_outliers)/len(selection)*100, 2)}%')
    return high_outliers + low_outliers