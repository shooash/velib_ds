import datetime
from typing import Union
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from general import log

class VelibTransformerDefaults:
    params = {
        'encode_time' : [
            ('hour', 24),
            ('weekday', 7),
            ], # liste de tuples/listes ('colname', periods)
        'geoencode' : True, # encoder lat et lon
        'hotencode' : ['cluster'],
        'lags' : 0, # int: nombre de cols lag à ajouter WIP
        'lags_cols' : [], # les colonnes lags WIP
        'scale' : ['capacity', 'temp', 'precip', 'gel', 'vent', 'conlat', 'conlon'], # cols à scaler
        'nonscale' : ['weekend', 'holiday', 'preholiday', 'postholiday', 'pont', 'vacances', 'vacances_uni'], # cols sans scaling et encoding
        'target' : 'delta',
        'smoothen' : True, # Lissage activé
        'smooth_window' : 3, # taille de fenetre de lissage: 3, 5, 7...
        'low_limit' : None, # Date minimale pour train
        'split_date' : None, # Date marge de train et test
        'cut_off_date' : None, # Date supérieur de données de teste
        'reconstructed' : True, # Garder les données reconstruite ou non
        'clusterize' : True,
        'lagged_value' : True, # last similar value known (station, weekday, hour, dayoff)
        'lagged_value_3hours_mean' : True, # last similar value known (station, weekday, hour, dayoff) rolling mean per 3 hours
        'mean_value' : True, # Mean for station, weekday, dayoff, hour (last 8 weeks)
        'mean_peaks' : True, # Mean for 6-12 hours cumulated flow for station, weekday (last 8 weeks)
    }
    
class VelibTransformer:
    """
    Secondary transformer class for training and prediction.
    transformation_params = {....}
    df = pd.read_hdf('processed.h5')....
    df = VelibTransformer.smoothen(df, window=3)
    df, df_train, df_test = VelibTransformer.split(df, transformation_params)
    transformer = VelibTransformer(transformation_params)
    df_train = transformer.fit_transform(df_train)
    df_test = transformer.transform(df_test)
    joblib.dump(transformer, 'transformer.joblib')
    """
    def __init__(self, params = {}):
        self.params = VelibTransformerDefaults.params.copy() | params
        # sauvegardons la liste de features actuelles pour info
        self.features = self.params['scale'] + self.params['nonscale'] + [self.params['target']]
        self.added_columns = []
        self.DELTA_TOP_MARGE = None
        self.DELTA_BOTTOM_MARGE = None
        self.stations_clusters = None

    def fit(self, df : pd.DataFrame):
        df = df.copy()
        log(f'Fitting dataset with shape: {df.shape}')
        if self.params.get('clusterize'):
            log('Fitting VelibClusterizer...')
            self.clusterizer = VelibClusterizer()
            self.clusterizer.fit(df)
            # self.added_columns.extend(['cluster', 'metacluster'])
            log('VelibClusterizer fitted.')

        self.lag_params = {
            'lagged_value' : self.params.get('lagged_value'),
            'lagged_value_3hours_mean' : self.params.get('lagged_value_3hours_mean'),
            'mean_value' : self.params.get('mean_value'),
            'mean_peaks' : self.params.get('mean_peaks')
        }
        if any(self.lag_params.values()):
            log(f'Fitting VelibLagger with params: {self.lag_params}')
            self.lagger = VelibLagger(self.lag_params, scaled=True)
            self.lagger.fit(df)
            for k, v in self.lag_params.items():
                if not v:
                    continue
                match k:
                    case 'mean_peaks':
                        self.added_columns.extend(['mean_6_12', 'mean_16_22'])
                    case _:
                        self.added_columns.append(k)
            log('VelibLagger fitted.')

        if self.params.get('geoencode'):
            self.geoencoder = VelibGeoEncoder()
            self.geoencoder.fit(df)
            self.added_columns.extend(['lat', 'lon'])

        if self.params['encode_time']:
            cols, _ = zip(*self.params['encode_time'])
            self.added_columns.extend(['sin_' + c for c in cols])
            self.added_columns.extend(['cos_' + c for c in cols])

        if self.params.get('scale'):
            # Il nous faudra assurer le type float pour les colonnes en scale
            safe_types = dict(zip(self.params['scale'], [float] * len(self.params['scale'])))
            df = df.astype(safe_types)
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.params['scale']])
            
        if self.params['reconstructed']:
            self.added_columns.append('reconstructed')
        # Enlever les doublons
        self.features = list(set(self.features))
        # Log marges
        self.log_marges(df)


    def transform(self, df : pd.DataFrame):
        df = df.copy()
        log('Transforming dataframe...')
        if self.params.get('clusterize'):
            df = self.clusterizer.transform(df)
        if any(self.lag_params.values()):
            df = self.lagger.transform(df)

        log('Encoding and scaling.')

        if self.params['encode_time']:
            cols, periods = zip(*self.params['encode_time'])
            df = VelibTransformer.encode_time(df, list(cols), list(periods))

        if self.params.get('geoencode'):
            df = self.geoencoder.transform(df)

        if self.params['hotencode']:
            if not hasattr(self, 'onehotencoder'):
                self.onehotencoder = OneHotEncoder(sparse_output=False)
                self.onehotencoder.fit(df[self.params['hotencode']])
            old_features = df.columns
            df.loc[:, self.onehotencoder.get_feature_names_out(self.params['hotencode'])] = self.onehotencoder.transform(df[self.params['hotencode']])
            new_features = [c for c in df.columns if c not in old_features]
            # self.drop_features(self.params['hotencode'])
            # self.add_features(new_features)
            self.added_columns.extend(new_features)
        
        if self.params.get('scale'):
            # Il nous faudra assurer le type float pour les colonnes en scale
            safe_types = dict(zip(self.params['scale'], [float] * len(self.params['scale'])))
            df = df.astype(safe_types)
            df.loc[:, self.params['scale']] = self.scaler.transform(df[self.params['scale']])

        # End by reconstucted drop
        if not self.params['reconstructed']:
            len_before = len(df)
            df = df.dropna()
            df = df[df['reconstructed'] == False]
            len_after = len(df)
            log(f'Removed {len_before-len_after} rows on reconstructed and NA cleaning, or {(len_before-len_after)/len_before*100:.2f}%')

        df = df[self.features + self.added_columns].copy()

        # Change all types to float
        # int_features = df.select_dtypes(include=['int', 'int16', 'int32', 'int64', 'integer'])
        int_features = [c for c in df.columns if c not in self.params.get('nonscale', []) + [self.params['target']]]
        float_types = dict(zip(int_features, [float] * len(int_features)))
        df = df.astype(float_types)

        if df.isna().sum().sum() != 0:
            raise RuntimeError("There're missing values in a dataset after transformation!")
        log(f'VelibTransformer final shape: {df.shape}')
        return df


    def fit_transform(self, df : pd.DataFrame):
        self.fit(df)
        return self.transform(df)


    def drop_features(self, cols: list[str]):
        '''
        Filter features list.
        '''
        self.features = [f for f in self.features if f not in cols]

    def add_features(self, cols: list[str]):
        '''
        Add new features to features list.
        '''
        self.features += [c for c in cols if c not in self.features]

    def replace_features(self, cols : list[str], prefix : str):
        '''
        Replace features in features list.
        '''
        # remove old features from list
        self.drop_features(cols)
        # add new features
        self.add_features([prefix + f for f in cols])

    # def hotencode(self, df: pd.DataFrame, cols: Union[str, list[str]]):
    #     if isinstance(cols, str):
    #         cols = [cols]
    #     orig_features = df.columns.to_list()
    #     df = pd.get_dummies(df, prefix=cols, columns=cols, dtype=int)
    #     # if hasattr(self, 'new_features'):
    #     #     # remplissons les cols inexistantes dans test avec les 0
    #     #     for col in self.new_features:
    #     #         if col not in df.columns:
    #     #             df[col] = 0
    #     # else:
    #     #     # Mise à jour de la liste de features
    #     self.new_features = [f for f in df.columns.to_list() if f not in orig_features]
    #     self.drop_features(self.params['hotencode'])
    #     self.add_features(self.new_features)
    #     return df

    def log_marges(self, df: pd.DataFrame):
        '''
        Save outliers limits to DELTA_TOP_MARGE and DELTA_BOTTOM_MARGE.
        '''
        Q1 = df['delta'].quantile(0.25)
        Q2 = df['delta'].quantile(0.75)
        IQR = Q2 - Q1
        self.DELTA_TOP_MARGE = Q2 + 1.5 * IQR
        self.DELTA_BOTTOM_MARGE = Q1 - 1.5 * IQR

    def save(self, filename : str):
        if self.lagger:
            self.lagger.prune()
        joblib.dump(self, filename)
    
    @staticmethod
    def load(filename : str):
        transformer = joblib.load(filename)
        if not isinstance(transformer, VelibTransformer):
            raise ValueError(f'Wrong object type loaded from file {filename}')
        return  transformer
    
    @staticmethod
    def encode_time(df: pd.DataFrame, cols: Union[str, list[str]], periods : int | list | None = None):
        '''
        Encoder les heures, jours, mois etc en représentation sin(x) - cos(x) pour la continuité de valeurs
        df: Pandas dataframe
        cols: Nom d'une ou des colonnes (str ou list de str)
        periods: Combien de périodes dans une saisonalité (eg 24 pour heures). Optionnel. Si pas précisé, sera calculé pas nombre de valeurs uniques.
        Resultat: colonnes sin_COLNAME, cos_COLNAME ajouté en df d'origine qui est retourné.
        '''
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(periods, int) or periods is None:
            periods = [periods]
        for i, c in enumerate(cols):
            # Si on a la périodicité pour cette colonne, ok, sinon c'est None
            if i < len(periods):
                p = periods[i]
            else:
                p = None
            # Si la périodicité et None, calculons à la base de nombre de valeurs uniques
            if p is None:
                p = df[c].nunique()
            df['sin_' + c] = np.sin(df[c] / p * 2 * np.pi)
            df['cos_' + c] = np.cos(df[c] / p * 2 * np.pi)
        return df

    @staticmethod
    def smoothen(df: pd.DataFrame, window = 3):
        '''
        Smoothen target data.
        '''
        log(f'Smoothening data {window=}...')
        df = df.copy()
        if window == 0:
            return df
        df['delta'] = (
            df.groupby(['station'], group_keys=False)[['delta', 'datehour']]
                    .apply(lambda x: x.rolling(window=f'{window}h', on='datehour', center=True, min_periods=1).mean())['delta']
                )
        log('Data smoothened.')
        return df

    @staticmethod
    def split(df : pd.DataFrame, params : dict = {}):
        """Cut off unneeded dates and split by date to train, test if split_date is set."""
        params = VelibTransformerDefaults.params.copy() | params
        log(f"Splitting data: {params['low_limit']} - {params['split_date']} - {params['cut_off_date']}")
        if isinstance(params['low_limit'], datetime.datetime):
            df = df[df.datehour.dt.date >= params['low_limit'].date()]
        if isinstance(params['cut_off_date'], datetime.datetime):
            df = df[df.datehour.dt.date < params['cut_off_date'].date()]
        if isinstance(params['split_date'], datetime.datetime):
            df_train : pd.DataFrame | None = df[df.datehour.dt.date < params['split_date'].date()].copy()
            df_test : pd.DataFrame | None = df[df.datehour.dt.date >= params['split_date'].date()].copy()
        else:
            log('No df_train and df_test to separate.')
            df_train = None
            df_test = None
        log(f'Split completed.')
        return df.copy(), df_train.copy(), df_test.copy()

class VelibGeoEncoder:
    def __init__(self, scale = True):
        self.scale = scale
        self.scaler = None
        
    def fit(self, df : pd.DataFrame):
        self.min_lat = df['lat'].min()
        self.min_lon = df['lon'].min()
    
    def transform(self, df : pd.DataFrame):
        df = VelibGeoEncoder.encode_lat(df, ['lat'], min_val=self.min_lat)
        df = VelibGeoEncoder.encode_lon(df, ['lon'], min_val=self.min_lon)
        if self.scale:
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(df[['lat', 'lon']])
            df.loc[:, ['lat', 'lon']] = self.scaler.transform(df[['lat', 'lon']])
        return df

    @staticmethod
    def encode_lat(df: pd.DataFrame, cols: Union[str, list[str]], min_val : dict | float | None = None):
        '''
        Encode latitude to km wise grid with corrected 0 point.
        '''
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(min_val, (int, float)):
            min_val = {c : float(min_val) for c in cols}

        for c in cols:
            minimum = min_val.get(c)
            if minimum is None:
                minimum = df[c].min()
            df[c] = (df[c] - minimum) * 111000
        return df


    @staticmethod
    def encode_lon(df: pd.DataFrame, cols: Union[str, list[str]], min_val : dict | float | None = None):
        '''
        Encode longitude to km wise grid with corrected 0 point.
        '''
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(min_val, (int, float)):
            min_val = {c : float(min_val) for c in cols}

        for c in cols:
            minimum = min_val.get(c)
            if minimum is None:
                minimum = df[c].min()
            df[c] = (df[c] - minimum) * 72987
        return df
        

class VelibLaggerDefaults:
    params = {
        'lagged_value' : True, # last similar value known (station, weekday, hour, dayoff)
        'lagged_value_3hours_mean' : True, # last similar value known (station, weekday, hour, dayoff) rolling mean per 3 hours
        'mean_value' : True, # Mean for station, weekday, dayoff, hour (last 8 weeks)
        'mean_peaks' : True, # Mean for 6-12 hours cumulated flow for station, weekday (last 8 weeks)
                             # AND Mean for 16-22 hours cumulated flow for station, weekday (last 8 weeks)        
    }
    
class VelibLagger:
    def __init__(self, params : dict = {}, scaled = True):
        self.params = VelibLaggerDefaults.params.copy() | params
        self.scaled = scaled

    def fit(self, df : pd.DataFrame):
        df = df.copy()
        # ### Use only last 12 weeks
        # df = df[df['datehour'] >= datetime.datetime.now() - datetime.timedelta(days=7 * 12 + 1)]
        log('Fitting VelibLagger')
        self.lag_columns = ['lagged_value']
        if 'date' not in df.columns:
            df['date'] = df['datehour'].dt.date
        df = df[['datehour', 'date', 'hour', 'station', 'weekend', 'weekday', 'holiday', 'pont', 'delta']]
        df['dayoff'] = df['holiday'] | df['pont']
        
        if self.params.get('mean_value'):
            log('Calculating mean_value')
            df['mean_value'] = df.groupby(['station', 'weekday', 'hour'], group_keys=False)['delta'].apply(lambda g: g.rolling(8, min_periods=3, center=False).mean().shift(1))
            self.lag_columns.append('mean_value')

        if self.params.get('mean_peaks'):
            log('Calculating mean_peaks')
            means_df = self.__get_station_weekday_sums_means(df)
            df = df.merge(means_df, on=['station', 'date'], how='left')
            self.lag_columns.extend(['mean_6_12', 'mean_16_22'])

        df = df.rename(columns={'delta' : 'lagged_value'})

        if self.params.get('lagged_value_3hours_mean'):
            log('Calculating lagged_value_3hours_mean')
            df['lagged_value_3hours_mean'] = df.groupby(['station'], group_keys=False)[['lagged_value', 'datehour']].apply(lambda g: g.rolling('3h', on='datehour', center=True).mean())['lagged_value']
            self.lag_columns.append('lagged_value_3hours_mean')

        log('Lags calculated for fitting.')
        self.data = df[['datehour', 'station', 'weekday', 'dayoff', 'hour'] + self.lag_columns]
        if self.scaled:
            self.__scale_fit()

    def transform(self, df : pd.DataFrame):
        log('VelibLagger transformation.')
        df = df.copy()
        df.sort_values('datehour')
        export_columns = list(df.columns)
        df['orig'] = True
        df['dayoff'] = df['holiday'] | df['pont']
        joint_df = df.merge(self.data, on=['datehour', 'station', 'weekday', 'dayoff', 'hour'], how='outer')
        
        if self.params.get('lagged_value'): # last similar value known (station, weekday, hour, dayoff)
            joint_df['lagged_value'] = joint_df.groupby(['station', 'weekday', 'dayoff', 'hour'])['lagged_value'].ffill().bfill()
            export_columns.append('lagged_value')
        if self.params.get('lagged_value_3hours_mean'):
            joint_df['lagged_value_3hours_mean'] = joint_df.groupby(['station', 'weekday', 'dayoff', 'hour'])['lagged_value_3hours_mean'].ffill().bfill()
            export_columns.append('lagged_value_3hours_mean')
        if self.params.get('mean_value'):
            joint_df['mean_value'] = joint_df.groupby(['station', 'weekday', 'dayoff', 'hour'])['mean_value'].ffill().bfill()
            export_columns.append('mean_value')
        if self.params.get('mean_peaks'):
            joint_df[['mean_6_12', 'mean_16_22']] = joint_df.groupby(['station', 'weekday'])[['mean_6_12', 'mean_16_22']].ffill().bfill()
            export_columns.extend(['mean_6_12', 'mean_16_22'])
        df = joint_df[joint_df['orig'].fillna(False).astype(bool)][export_columns]
        if self.scaled:
            df = self.__scale_transform(df)
        return df    
    
    def fit_transform(self, df : pd.DataFrame):
        self.fit(df)
        return self.transform(df)
    
    def prune(self):
        """For saving purposes: keep only last 8 weeks of data"""
        max_date = self.data['datehour'].max()
        cutoff_date = max_date - datetime.timedelta(days=57)
        self.data = self.data[self.data['datehour'] > cutoff_date]
        
    def __get_station_weekday_sums_means(self, velib_data : pd.DataFrame, keep_incomplete = False):
        '''
        Get the average for summarized flow for peak hours from 5 to 12 and from 15 to 22 hours.
        '''
        # sum_or_nan va retourner NaN pour toutes les series avec des NAs
        aggregation_task = {
            'delta6h12h' : pd.NamedAgg(column='delta', aggfunc=lambda x: sum_or_nan(x.iloc[5:12])),
            'delta16h22h' : pd.NamedAgg(column='delta', aggfunc=lambda x: sum_or_nan(x.iloc[15:22])),
            'weekday' : pd.NamedAgg('weekday', 'first')
        }
        stations_weekdays_peaks_means = velib_data.groupby(['station', 'date']).agg(**aggregation_task).reset_index()
        # stations_weekdays_peaks_means['date'] = pd.to_datetime(stations_weekdays_peaks_means['date'])
        # On ne veut pas calculer les journées non complètes
        if not keep_incomplete:
            stations_weekdays_peaks_means = stations_weekdays_peaks_means.dropna()
        # aggregation_task_2 = {
        #         'delta6h12h' : 'mean',
        #         'delta16h22h' : 'mean',
        #     }
        # stations_weekdays_peaks_means = (stations_weekdays_peaks_means 
        #             .groupby(['station', 'weekday'])
        #                 .rolling(8, center=False) # Count only last 8 weeks
        #                     .agg(aggregation_task_2))
        stations_weekdays_peaks_means[['mean_6_12', 'mean_16_22']] = (
            stations_weekdays_peaks_means
                .groupby(['station', 'weekday'])[['delta6h12h', 'delta16h22h']]
                    .transform(lambda g: g.shift(1).rolling(8, min_periods=4, center=False).mean())
            )
        return stations_weekdays_peaks_means[['station', 'date', 'mean_6_12', 'mean_16_22']].dropna()
    
    def __scale_fit(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data[self.lag_columns])
        
    def __scale_transform(self, df : pd.DataFrame):
        df.loc[:, self.lag_columns] = self.scaler.transform(df[self.lag_columns])
        return df
        
class VelibClusterizer:
    def __init__(self, clusters_range : list[int] = [25, 125], region_range : list[int] = [10, 20], debug=True):
        self.clusters_range = clusters_range
        self.region_range = region_range
        self.debug = debug
    
    def fit(self, df : pd.DataFrame, keep_incomplete : bool = False):
        df = df.copy()
        ### Use only last 4 weeks for clusters 
        cut_off_date = df['datehour'].max() - datetime.timedelta(days=7 * 4 + 1)
        df = df[df['datehour'] >= cut_off_date]
        log('Calculating stations statistics for clusterizing')
        stations = self.__get_station_days_stats(df, keep_incomplete=keep_incomplete)
        log('Converting lat and lon to km position')
        stations['lat'] = stations.lat.apply(float)
        stations['lon'] = stations.lon.apply(float)
        min_lon = stations.lon.min()
        min_lat = stations.lat.min()
        stations['convlon'] = stations.lon.apply(convert_lon, args=[min_lon]) # type: ignore
        stations['convlat'] = stations.lat.apply(convert_lat, args=[min_lat]) # type: ignore
        log('Calculating best silhouette score for clusters')
        self.n_clusters = self._get_best_silhouette_score(stations, 
                    ['convlat', 'convlon', 'delta6h12h', 'delta16h22h'], 
                    min_c=self.clusters_range[0], 
                    test_size=self.clusters_range[1] - self.clusters_range[0], 
                    debug=self.debug).astype(int)
        if self.debug:
            log(f'Calcule de {self.n_clusters} clusters')
        model = KMeans(n_clusters=self.n_clusters, random_state=0) if self.debug else KMeans(n_clusters=self.n_clusters)
        kmeans = model.fit(stations[['convlat', 'convlon', 'delta6h12h', 'delta16h22h']])
        stations['cluster'] = kmeans.labels_
        
        ### Ajoutons des meta clusters pour expérimentation d'apprentissage séparée
        log('Calculating best silhouette score for regions')
        self.regions_n_clusters = self._get_best_silhouette_score(stations, 
                    ['convlat', 'convlon', 'delta6h12h', 'delta16h22h'], 
                    min_c=self.region_range[0], 
                    test_size=self.region_range[1] - self.region_range[0],
                    debug=self.debug).astype(int)
        if self.debug:
            log(f'Calcule de {self.regions_n_clusters} meta clusters')
        model = KMeans(n_clusters=self.regions_n_clusters, random_state=0) if self.debug else KMeans(n_clusters=self.regions_n_clusters)
        kmeans = model.fit(stations[['convlat', 'convlon', 'delta6h12h', 'delta16h22h']])
        stations['metacluster'] = kmeans.labels_
        self.clusters_df = stations[['station', 'cluster', 'metacluster']]

    def transform(self, df : pd.DataFrame):
        log('VelibCluster transformation.')
        return pd.merge(df.copy(), self.clusters_df.copy(), on='station', how='left')
    
    def fit_transform(self, df : pd.DataFrame, keep_incomplete : bool = False):
        self.fit(df, keep_incomplete)
        return self.transform(df)

    def _get_best_silhouette_score(self, stations, cols : list, min_c = 50, debug = False, test_size=100):
        silhouette_scores = []
        for k in range(min_c, min_c + test_size + 1):
            if debug: log(f'Calculating silhouet score for clusters {k - min_c} of {test_size}...', end='\r')
            kmeans = KMeans(n_clusters=k, random_state=0).fit(stations[cols])
            silhouette_scores.append(silhouette_score(stations[cols], kmeans.labels_))
        best_score = np.max(silhouette_scores)
        best_cluster_num = np.argmax(silhouette_scores) + min_c
        if debug: log(f'Calculated all clusters, best score is {best_score} for {best_cluster_num} clusters')
        return best_cluster_num


    def __get_station_days_stats(self, velib_data : pd.DataFrame, keep_incomplete = False):
        '''
        Get the average for summarized flow for peak hours from 5 to 12 and from 15 to 22 hours.
        '''
        # sum_or_nan va retourner NaN pour toutes les series avec des NAs
        aggregation_task = {
            'delta6h12h' : pd.NamedAgg(column='delta', aggfunc=lambda x: sum_or_nan(x.iloc[5:12])),
            'delta16h22h' : pd.NamedAgg(column='delta', aggfunc=lambda x: sum_or_nan(x.iloc[15:22])),
        }
        if 'lat' in velib_data.columns:
            aggregation_task['lat'] = pd.NamedAgg('lat', 'first')
        if 'lon' in velib_data.columns:
            aggregation_task['lon'] = pd.NamedAgg('lon', 'first')
             
        stations_days = velib_data.groupby(['station', 'date']).agg(**aggregation_task)
        # On ne veut pas calculer les journées non complètes
        if not keep_incomplete:
            stations_days = stations_days.dropna()
        aggregation_task_2 = {
                'delta6h12h' : 'mean',
                'delta16h22h' : 'mean',
            }
        if 'lat' in stations_days.columns:
            aggregation_task_2['lat'] = 'first'
        if 'lon' in stations_days.columns:
            aggregation_task_2['lon'] = 'first'

        stations_days = stations_days.groupby(['station']).agg(aggregation_task_2).reset_index()
        return stations_days

def sum_or_nan(values : pd.Series):
    '''
    Summarize only sequences without NA. Else return NA.
    '''
    if values.isna().any():
        return np.nan
    else:
        return values.sum()

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

