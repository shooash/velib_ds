import copy
import datetime
from time import sleep
from typing import Literal, Tuple
import mlflow.data.filesystem_dataset_source
import mlflow.data.pandas_dataset
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import mlflow
# from .file_targets import LOGS_FOLDER, FULL_DATASET, DataFiles.train, DataFiles.test, PRED_DATASET
# from .file_targets import TRANSFORMER_TRAIN_TEST, DataFiles.processed, TRANSFORMER_RELEASE, TRANSFORMER_PROD
import joblib
from pathlib import Path
import logging 
# from .data import import_dataset, save_dataset, load_or_import_pred_dataset
# from .connect import get_run_id_from_file
import os
from typing import Literal, Tuple
import pandas as pd
# from tensorflow import keras # type: ignore
import keras
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as MAE, root_mean_squared_error as RMSE
from general import DataFiles, Storage


MLFlow_URI = os.environ.get('MLFLOW_TRACKING_URI') or 'http://127.0.0.1:8080'

class BaseFlow:
    COLS_BLACKLIST = ['metacluster', 'cluster', 'station', 'datehour']
    X_train : pd.DataFrame | None
    y_train : pd.Series | pd.DataFrame | None
    df_test : pd.DataFrame | None
    y_test : pd.Series | pd.DataFrame | None
    # ID de la station Chatelet
    chatelet  = '82328045'
    def __init__(self, name : str, baselib : Literal['sklearn', 'keras', 'tensorflow'] = 'sklearn', *, mlflow_uri=MLFlow_URI, mode : Literal['test', 'release', 'prod'] = 'test') -> None:
        self.name = name
        self.mlflow_uri = mlflow_uri
        self.metrics = {}
        self.params = {}
        self.model = None
        self.input_example = None
        self.baselib = baselib
        self.mode = mode
        self.model_filename = None
        self.init_log()
        self.logger.info('Created instance for ' + self.name)
        self.X_train = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_pred = None
        self.input_example = None
        self.dataid = None
        if self.mlflow_uri is not None:
            self.init_mlflow()

    def init_mlflow(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.name)

    def set_params(self, **params) -> None:
        self.params = params
        self.logger.debug('Params set to: ' + str(params))
    
    def update_params(self, **params) -> None:
        # enforce types like wh = float('1')
        params = {k : type(self.params[k])(v) for k, v in params.items() }
        self.params = self.params | params
        self.logger.debug('Params updated with: ' + str(params))
        self.logger.debug('Current params: ' + str(self.params))

    def fit(self, df : pd.DataFrame, y : pd.Series):
        self.X_train = df
        self.y_train = y
        self.log_marges(y)
        self.set_mlflow_run('fit')

    def predict(self, df : pd.DataFrame) -> None:
        if not len(df):
            raise RuntimeError('Prediction dataframe is empty.')
        self.set_mlflow_run('predict')


    def score(self) -> None:
        raise NotImplementedError()

    def save_model(self, basename : str) -> str:
        '''Return FileName'''
        raise NotImplementedError()
    
    def with_dataset(self, df : pd.DataFrame):
        if 'datehour' not in df.columns:
            return self
        first_dt = df['datehour'].min().strftime('%Y%m%d_%H')
        last_dt = df['datehour'].max().strftime('%Y%m%d_%H')
        self.dataid = f'{first_dt}-{last_dt}_train'
        return self
        
    def set_mlflow_run(self, starter = 'default'):
        if self.mlflow_uri is None or mlflow.active_run() is not None:
            return
        mlflow.start_run(tags={'starter' : starter})
        mlflow.log_params(self.params)
        if self.X_train is not None:
            if not self.dataid:
                self.with_dataset(self.X_train)
            dataset = mlflow.data.pandas_dataset.from_pandas(self.X_train.iloc[-min(100, len(self.X_train)):], name=self.dataid)
            mlflow.log_input(dataset, context=starter)
        # if self.df_test is not None and self.X_test is not None:
        #     first_dt = self.df_test['datehour'].min().strftime('%Y%m%d_%H')
        #     last_dt = self.df_test['datehour'].max().strftime('%Y%m%d_%H')
        #     dataset = mlflow.data.pandas_dataset.from_pandas(self.X_test.iloc[0:min(100, len(self.X_test))], name=f'{first_dt}-{last_dt}_test')
        #     mlflow.log_input(dataset, context=self.mode)
        # Enable mlflow autolog
        mlflow.tensorflow.autolog(log_models=False, registered_model_name=self.name, log_every_epoch=True, checkpoint=False, log_input_examples=True, log_datasets=False) # type: ignore


    def save_mlflow(self, base_name : str):
        self.set_mlflow_run('save')
        mlflow.log_metrics(self.metrics)
        run_name = mlflow.active_run().info.run_name #type: ignore
        mlflow.end_run()
        return run_name

    def end(self):
        self.save_mlflow()

    def save(self, save_model_name = None) -> None:
        base_name = '_'.join(
            [self.name] + 
            [f'{k}-{v}' for k,v in self.params.items()]
            )
        self.model_filename = self.save_model(save_model_name or base_name)
        self.logger.info('Saved model to ' + self.model_filename)
        if self.mlflow_uri is not None:
            run_name = self.save_mlflow(base_name)
            self.logger.info(f'Saved MLFlow run {run_name}')
        self.logger.debug('All saved.')

    def init_log(self):
        # Create logger
        self.logger = logging.getLogger('dual_logger')
        self.logger.setLevel(logging.DEBUG)  # Set the logging level
        if self.logger.hasHandlers():
            return
        file_handler = logging.FileHandler(Storage.logs(f'{self.name}.log'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def params_from_run_id(self, run_id: str):
        try:
            run = mlflow.get_run(run_id)
        except:
            raise RuntimeError(f'Unable to get run with {run_id=}.')
        params = {k : v for k, v in run.data.params.items() if k in self.params}
        self.update_params(**params)
        return self

    def with_run(self, run_id = 'best'):
        '''
        Run Flow with id from local/params/{run_id}.run.txt.
        '''
        # run_id = get_run_id_from_file(run_id)
        if not run_id:
            raise RuntimeError(f'To train best model on full data put run_id to local/params/{run_id}.run.txt.')
        return self.params_from_run_id(run_id)

    def log_marges(self, y: pd.Series):
        '''
        Save outliers limits to DELTA_TOP_MARGE and DELTA_BOTTOM_MARGE.
        '''
        Q1 = y.quantile(0.25)
        Q2 = y.quantile(0.75)
        IQR = Q2 - Q1
        self.DELTA_TOP_MARGE = Q2 + 1.5 * IQR
        self.DELTA_BOTTOM_MARGE = Q1 - 1.5 * IQR
    

class MLPFlow(BaseFlow):

    def __init__(self, mode : Literal['test', 'release', 'prod'] = 'test') -> None:
        super().__init__('MLP_flow', baselib='keras', mode=mode)
        self.embedding = {}
        # self.stations_train = self.X_train['station'] if self.X_train is not None else None
        # self.clusters_train = self.X_train['cluster'] if self.X_train is not None and 'cluster' in self.X_train.columns else None
        # if self.df_test is not None:
        #     self.stations_test = self.df_test['station']
        #     self.clusters_test = self.df_test['cluster']
        # else:
        #     self.stations_test = None
        #     self.clusters_test = None
        self.history = None
        self.params = {
            'wl' : 3.0,
            'wh' : 3.0,
            'loss_fn' : 'wmse',
            'batch' : 128000,
            'opt' : 'adam',
            'emb' : 'stations+clusters',
            'act' : 'relu',
            'epochs' : 200,
        }


    def compile(self) -> None:
        if self.X_train is None:
            raise RuntimeError('Unable to compile a model without train dataset.')
        features = [c for c in self.X_train if c not in self.COLS_BLACKLIST]
        features_size = len(features)
        act = self.params['act']
        embedding = self.params['emb']
        # inputs_list = []
        inputs_dict = {}
        embeddings_list = []
        # Station Embedding
        if embedding:
            for e, e_list in self.embedding.items():
                e_input = keras.layers.Input(shape=(1,), name=e)
                e_embed = keras.layers.Embedding(input_dim=len(e_list), output_dim=min(50, round(len(e_list)**0.4)))(e_input)
                e_embed = keras.layers.Flatten()(e_embed)
                inputs_dict[e] = e_input
                embeddings_list.append(e_embed)
        feature_input = keras.layers.Input(shape=(features_size,), name='features')
        inputs_dict['features'] = feature_input

        x = keras.layers.Concatenate(axis=-1)(embeddings_list + [feature_input])
        x = keras.layers.Dense(64, activation=act)(x)
        x = keras.layers.Dropout(0.1)(x)
        # x = Dense(48, activation=act)(x)
        # x = Dropout(0.1)(x)
        x = keras.layers.Dense(32, activation=act)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(16, activation=act)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(8, activation=act)(x)
        output = keras.layers.Dense(1)(x)

        # Custom loss functions:
        @keras.saving.register_keras_serializable()
        def wmae(y_true, y_pred):
            error = tf.abs(y_true - y_pred)
            weights = tf.where(
                y_true > self.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)
        @keras.saving.register_keras_serializable()
        def wmse(y_true, y_pred):
            error = tf.square(y_true - y_pred)
            weights = tf.where(
                y_true > self.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)
        
        match self.params['opt']:
            case 'sgd':
                optimizer = keras.optimizers.SGD(momentum=0.5)
            case 'adamw':
                optimizer = keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4)
            case _:
                optimizer = self.params['opt']
        
        match self.params['loss_fn']:
            case 'wmae':
                loss = wmae
                metrics = 'mae'
            case 'wmse':
                loss = wmse
                metrics = 'mse'
            case 'mae':
                loss = 'mae'
                metrics = 'mse'
            case 'mse':
                loss = 'mse'
                metrics = 'mae'
            case _:
                loss = self.params['loss_fn']
                metrics = 'mse'

        # Input Example pour MLFlow register
        # self.input_example = inputs_list + [feature_input] if inputs_list else feature_input

        self.model = keras.models.Model(
            inputs = inputs_dict,
            # inputs=inputs_list + [feature_input] if inputs_list else feature_input, 
            outputs=output
            )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics]) #type: ignore
        self.model.summary()

    def fit(self, df : pd.DataFrame, y : pd.Series):
        super().fit(df, y)
        # Create embedding dict = {'station' : {'1234' : 0, '5678' : 1}}
        if self.params.get('emb'):
            for e in self.params.get('emb'):
                self.embedding[e] = {k : i for i, k in enumerate(df[e].unique())}
        self.logger.debug('Compiling the model with params: ' + str(self.params))
        self.compile()
        self.logger.debug('Training model with params: ' + str(self.params))
        # Filter embedded items
        columns = [c for c in df.columns if c not in self.COLS_BLACKLIST]
        # Prepare inputs to fit
        inputs_dict = {}
        # Station Embedding
        for e, e_dict in self.embedding.items():
            inputs_dict[e] = df[e].map(lambda x: e_dict.get(x, -1))
        inputs_dict['features'] = df[columns].values.astype(float)
        self.history = self.model.fit( 
            inputs_dict,
            self.y_train,
            epochs = self.params['epochs'],
            batch_size = self.params['batch'],
            validation_split = 0.3,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                ],
            verbose=2 #type: ignore
            )
        self.logger.info('Model compiled and fitted.')
        return self

    # def test(self):
    #     super().test()
    #     self.logger.debug('Testing model...')
    #     if self.model is None:
    #         raise RuntimeError('Model not defined.')
    #     if self.X_test is None:
    #         raise RuntimeError('X_test not defined.')
    #     embedding = self.params['emb']
    #     inputs_dict={}
    #     inputs_dict['features'] = self.X_test.values
    #     # Station Embedding
    #     if 'station' in embedding and self.stations_test is not None:
    #         inputs_dict['station'] = self.stations_test.values
    #     if 'cluster' in embedding and self.clusters_test is not None:
    #         inputs_dict['cluster'] = self.clusters_test.values
    #     y_pred = self.model.predict(
    #         inputs_dict,
    #         # inputs_list + [self.X_test] if inputs_list else self.X_test, 
    #         batch_size=self.params['batch'], verbose=2) #type: ignore
    #     self.y_pred = y_pred.flatten().astype(float)
    #     return self
    
    def set_model(self,  model):
        self.model = model
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        '''
        Predict used preloaded model and params.
        '''
        super().predict(df)
        self.logger.debug('Predicting...')
        if self.model is None:
            raise RuntimeError('Model not defined.')
        # Filter embedded items
        columns = [c for c in df.columns if c not in self.COLS_BLACKLIST]
        # Prepare inputs to fit
        inputs_dict = {}
        # Station Embedding
        for e, e_dict in self.embedding.items():
            inputs_dict[e] = df[e].map(lambda x: e_dict.get(x, -1))
        inputs_dict['features'] = df[columns].values.astype(float)
        y_pred = self.model.predict(
            inputs_dict,
            batch_size=self.params['batch'], verbose=2) #type: ignore
        self.y_pred = y_pred.flatten().astype(float)
        return self.y_pred

    # Metrics
    def get_mare(self):
        '''Get Mean Absolute Rush Error i.e. MAE for sums of flow at 6-12 and 15-22 per day-station'''
        if self.df_test is None or self.y_test is None:
            raise RuntimeError('No scoring without X_test available')
        df = self.df_test[self.df_test.hour.between(6, 12) | self.df_test.hour.between(15, 22)].copy()
        df['morning'] = df.hour <= 12
        # Count total flow for morning and evening rush hours per day-station 
        scoring = df.groupby(['station_orig', 'date', 'morning'], as_index=False)[['delta_test', 'delta_pred']].sum()
        # Get absolute error for rush hours
        scoring['are'] = (scoring['delta_test'] - scoring['delta_pred']).abs()
        self.metrics['mare'] = float(scoring['are'].mean())
        self.metrics['chatelet_mare'] = float(scoring[scoring.station_orig.astype(str) == self.chatelet]['are'].mean())
        test_date = self.df_test[self.df_test.weekday == 3].datehour.dt.date.iloc[0]
        self.metrics['date_mare'] = float(scoring[scoring.date.dt.date == test_date]['are'].mean())
        self.metrics['date_chatelet_mare'] = float(scoring[(scoring.date.dt.date == test_date) & (scoring.station_orig.astype(str) == self.chatelet)]['are'].mean())
        
    def score(self):
        if self.df_test is None or self.y_test is None:
            raise RuntimeError('No scoring without X_test available')
        self.logger.debug('Calculating metrics.')
        self.df_test['delta_test'] = self.y_test
        self.df_test['delta_pred'] = self.y_pred
        self.get_mare()
        self.metrics['rmse'] = RMSE(self.y_test, self.y_pred)
        self.metrics['mae'] = MAE(self.y_test, self.y_pred)
        if self.history is not None:
            self.metrics['epochs'] = len(self.history.epoch)
        self.logger.info('Metrics calculated.')
        self.logger.info(self.metrics)
        return self

    @staticmethod
    def load(basename : str):
        print(f"Loading model from {Storage.models(basename + '.pkl')}")
        mlpflow = joblib.load(Storage.models(basename + '.pkl'))
        mlpflow.load_model(basename)
        return mlpflow
    
    @staticmethod
    def prune_s(obj : object):
        """Drop heavy objects that are not important for predictions before saving"""
        obj = copy.copy(obj)
        obj.X_train = None
        obj.X_test = None
        obj.y_train = None
        obj.history = None
        obj.model = None
        
        return obj
    
    def prune(self):
        """Drop heavy and unserializable objects that are not important for predictions before saving"""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.history = None
        self.model = None
                       
    def save_model(self, basename: str) -> str:
        self.model.save(Storage.models(basename + '.keras'))
        # self._save_stations_clusters(basename)
        # self.prune()
        joblib.dump(MLPFlow.prune_s(self), Storage.models(basename + '.pkl'))
        return Storage.models(basename + '.pkl')
        # return Storage.models(basename + '.keras')

    def load_model(self, basename : str):
        path = Storage.models(basename + '.keras')

        def wmae(y_true, y_pred):
            error = tf.abs(y_true - y_pred)
            weights = tf.where(
                y_true > self.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)

        def wmse(y_true, y_pred):
            global MLP_WEIGHTS
            error = tf.square(y_true - y_pred)
            weights = tf.where(
                y_true > self.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)

        return self.set_model(keras.saving.load_model(path, custom_objects={'Custom>wmae' : wmae, 'Custom>wmse' : wmse}))

class MLPFlowSeries(BaseFlow):
    
    def __init__(self, hours=12, mode : Literal['test', 'release', 'prod'] = 'test') -> None:
        super().__init__(f'MLP_Series_{hours}h', baselib='keras', mode=mode)
        self.hours = hours
        self.stations_train = self.X_train['station'] if self.X_train is not None else None
        self.clusters_train = self.X_train['cluster'] if self.X_train is not None else None
        if self.df_test is not None:
            self.stations_test = self.df_test['station']
            self.clusters_test = self.df_test['cluster']
        else:
            self.stations_test = None
            self.clusters_test = None
        self.history = None
        self.params = {
            'wl' : 3.0,
            'wh' : 3.0,
            'loss_fn' : 'mse',
            'batch' : 32000,
            'opt' : 'adam',
            'emb' : 'stations+clusters',
            'act' : 'relu',
            'epochs' : 200,
        }
        self.add_history()

    def add_history(self):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError('Unable to add history without train dataset.')
        if isinstance(self.y_train, pd.DataFrame):
            raise TypeError('Cannot add history if multiple y_train columns exist.')
        self.y_train = self.__add_history(self.X_train, self.X_train, self.y_train)
        self.stations_train = self.X_train['station']
        self.clusters_train = self.X_train['cluster']
        if isinstance(self.y_test, pd.Series) and isinstance(self.df_test, pd.DataFrame) and isinstance(self.X_test, pd.DataFrame):
            self.y_test = self.__add_history(self.df_test, self.X_test, self.y_test)
            self.stations_test = self.df_test['station']
            self.clusters_test = self.df_test['cluster']

    def __add_history(self, df : pd.DataFrame, X : pd.DataFrame, y : pd.Series) -> pd.DataFrame:
        '''Returns y_train : pd.DataFrame, stations_train : pd.Series'''
        df['delta_0'] = y
        X['delta_0'] = y
        for hour in range(1, self.hours + 1):
            X[f'lag_{hour}'] = df.groupby(['station'])['delta_0'].shift(hour)
            df[f'lag_{hour}'] = X[f'lag_{hour}']
            X[f'delta_{hour}'] = df.groupby(['station'])['delta_0'].shift(-hour)
            df[f'delta_{hour}'] = X[f'delta_{hour}']
        X.dropna(inplace=True)
        df.dropna(inplace=True)
        y_cols = [c for c in X.columns if 'delta_' in c]
        X.drop(columns=y_cols, inplace=True)
        return df[y_cols]

    def compile(self) -> None:
        if self.X_train is None:
            raise RuntimeError('Unable to compile a model without train dataset.')
        features_size = len(self.transformer.features) - 1 + self.hours
        act = self.params['act']
        embedding = self.params['emb']
        # inputs_list = []
        inputs_dict = {}
        embeddings_list = []
        # Station Embedding
        if 'station' in embedding and self.stations_train is not None:
            station_input =  keras.layers.Input(shape=(1,), name='station')
            station_embedding = keras.layers.Embedding(input_dim=self.stations_train.nunique(), output_dim=min(50, round(self.stations_train.nunique()**0.4)))(station_input)
            station_embedding = keras.layers.Flatten()(station_embedding)
            inputs_dict['station'] = station_input
            embeddings_list.append(station_embedding)
        # Cluster Embedding
        if 'cluster' in embedding and self.clusters_train is not None:
            cluster_input = keras.layers.Input(shape=(1,), name='cluster')
            cluster_embedding = keras.layers.Embedding(input_dim=self.clusters_train.nunique(), output_dim=min(50, round(self.clusters_train.nunique()**0.5)))(cluster_input)
            cluster_embedding = keras.layers.Flatten()(cluster_embedding)
            inputs_dict['cluster'] = cluster_input
            embeddings_list.append(cluster_embedding)
        feature_input = keras.layers.Input(shape=(features_size,), name='features')
        inputs_dict['features'] = feature_input
        # Combine Inputs
        # if inputs_list:
        x = keras.layers.Concatenate(axis=-1)(embeddings_list + [feature_input])
        # else:
        #     x = feature_input
        x = keras.layers.Dense(64, activation=act)(x)
        x = keras.layers.Dropout(0.1)(x)
        # x = Dense(48, activation=act)(x)
        # x = Dropout(0.1)(x)
        x = keras.layers.Dense(32, activation=act)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(16, activation=act)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(8, activation=act)(x)
        output = keras.layers.Dense(self.hours + 1)(x)

        # Custom loss functions:
        @keras.saving.register_keras_serializable()
        def wmae(y_true, y_pred):
            error = tf.abs(y_true - y_pred)
            weights = tf.where(
                y_true > self.transformer.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.transformer.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)
        @keras.saving.register_keras_serializable()
        def wmse(y_true, y_pred):
            global MLP_WEIGHTS
            error = tf.square(y_true - y_pred)
            weights = tf.where(
                y_true > self.transformer.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.transformer.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)
        
        match self.params['opt']:
            case 'sgd':
                optimizer = keras.optimizers.SGD(momentum=0.5)
            case 'adamw':
                optimizer = keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4)
            case _:
                optimizer = self.params['opt']
        
        match self.params['loss_fn']:
            case 'wmae':
                loss = wmae
                metrics = 'mae'
            case 'wmse':
                loss = wmse
                metrics = 'mse'
            case 'mae':
                loss = 'mae'
                metrics = 'mse'
            case 'mse':
                loss = 'mse'
                metrics = 'mae'
            case _:
                loss = self.params['loss_fn']
                metrics = 'mse'

        self.model = keras.models.Model(
            inputs = inputs_dict,
            # inputs=inputs_list + [feature_input] if inputs_list else feature_input, 
            outputs=output
            )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        self.model.summary()

    def train(self):
        super().train()
        self.logger.debug('Compiling the model with params: ' + str(self.params))
        self.compile()
        self.logger.debug('Training model with params: ' + str(self.params))
        embedding = self.params['emb']
        inputs_dict = {}
        inputs_dict['features'] = self.X_train.values
        # Station Embedding
        if 'station' in embedding and self.stations_train is not None:
            inputs_dict['station'] = self.stations_train.values
        if 'cluster' in embedding and self.clusters_train is not None:
            inputs_dict['cluster'] = self.clusters_train.values
        self.history = self.model.fit( 
            inputs_dict,
            self.y_train,
            epochs = self.params['epochs'],
            batch_size = self.params['batch'],
            validation_split = 0.3,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                ],
            verbose=2
            )
        self.logger.info('Model compiled and fitted.')
        return self

    def test(self):
        super().test()
        self.logger.debug('Testing model...')
        if self.model is None:
            raise RuntimeError('Model not defined.')
        embedding = self.params['emb']
        inputs_dict={}
        inputs_dict['features'] = self.X_test.values
        # Station Embedding
        if 'station' in embedding and self.stations_test is not None:
            inputs_dict['station'] = self.stations_test.values
        if 'cluster' in embedding and self.clusters_test is not None:
            inputs_dict['cluster'] = self.clusters_test.values
        y_pred = self.model.predict(
            inputs_dict,
            # inputs_list + [self.X_test] if inputs_list else self.X_test, 
            batch_size=self.params['batch'], verbose=2) #type: ignore
        self.y_pred = y_pred.astype(float)
        return self

    def predict(self, df: pd.DataFrame):
        '''
        Predict used preloaded model and params.
        '''
        super().predict(df)
        self.logger.debug('Predicting...')
        if self.model is None:
            raise RuntimeError('Model not defined.')
        embedding = self.params['emb']
        inputs_dict={}
        inputs_dict['features'] = self.X_test.values
        if 'station_orig' not in df.columns:
            df = self.transformer.get_stations_clusters(df)
        # Station Embedding
        if 'station' in embedding and self.stations_test is not None:
            inputs_dict['station'] = self.stations_test.values
        if 'cluster' in embedding and self.clusters_test is not None:
            inputs_dict['cluster'] = self.clusters_test.values
        # prediction
        y = self.model.predict(
            inputs_dict,
            batch_size=self.params['batch'], verbose=2)
        self.save_mlflow(self.name)
        return y.astype(float)
    
    # Metrics
    def get_mare(self):
        '''Get Mean Absolute Rush Error i.e. MAE for sums of flow at 6-12 and 15-22 per day-station'''
        if self.df_test is None or self.y_test is None:
            raise RuntimeError('No scoring without X_test available')
        df = self.df_test[self.df_test.hour.between(6, 12) | self.df_test.hour.between(15, 22)].copy()
        df['morning'] = df.hour <= 12
        # Count total flow for morning and evening rush hours per day-station 
        scoring = df.groupby(['station_orig', 'date', 'morning'], as_index=False)[['delta_test', 'delta_pred']].sum()
        # Get absolute error for rush hours
        scoring['are'] = (scoring['delta_test'] - scoring['delta_pred']).abs()
        self.metrics['mare'] = float(scoring['are'].mean())
        self.metrics['chatelet_mare'] = float(scoring[scoring.station_orig.astype(str) == self.chatelet]['are'].mean())
        test_date = self.df_test[self.df_test.weekday == 3].datehour.dt.date.iloc[0]
        self.metrics['date_mare'] = float(scoring[scoring.date.dt.date == test_date]['are'].mean())
        self.metrics['date_chatelet_mare'] = float(scoring[(scoring.date.dt.date == test_date) & (scoring.station_orig.astype(str) == self.chatelet)]['are'].mean())
        
    def score(self):
        if self.df_test is None or self.y_test is None:
            raise RuntimeError('No scoring without X_test available')
        self.logger.debug('Calculating metrics.')
        self.df_test['delta_test'] = self.y_test['delta_0']
        self.df_test['delta_pred'] = self.y_pred[:, 0]
        self.get_mare()
        self.metrics['rmse'] = RMSE(self.y_test, self.y_pred)
        self.metrics['mae'] = MAE(self.y_test, self.y_pred)
        if self.history is not None:
            self.metrics['epochs'] = len(self.history.epoch)
        self.logger.info('Metrics calculated.')
        self.logger.info(self.metrics)
        return self
    
    def set_model(self,  model):
        self.model = model
        return self

    def save_model(self, basename: str) -> str:
        self.model.save(Storage.models(basename + '.keras'))
        return Storage.models(basename + '.keras')

    def load_model(self, basename : str):
        path = Storage.models(basename + '.keras')

        def wmae(y_true, y_pred):
            error = tf.abs(y_true - y_pred)
            weights = tf.where(
                y_true > self.transformer.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.transformer.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)

        def wmse(y_true, y_pred):
            global MLP_WEIGHTS
            error = tf.square(y_true - y_pred)
            weights = tf.where(
                y_true > self.transformer.DELTA_TOP_MARGE, self.params['wh'],
                tf.where(y_true < self.transformer.DELTA_BOTTOM_MARGE, self.params['wl'], 1.0)
            )
            return tf.reduce_mean(error * weights)

        return self.set_model(keras.saving.load_model(path, custom_objects={'Custom>wmae' : wmae, 'Custom>wmse' : wmse}))
