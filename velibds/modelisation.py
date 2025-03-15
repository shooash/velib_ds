import pandas as pd, numpy as np, plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import BinaryCrossentropy, LogCosh, Huber
from tensorflow.keras.metrics import RootMeanSquaredError as KerasRMSE
from tensorflow.keras.optimizers import Optimizer, Adam, AdamW, Nadam
from tensorflow.keras.saving import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as RMSE, mean_absolute_error as MAE
from imblearn.over_sampling import SMOTE
import datetime, json, joblib
from .viz import VelibDataViz as viz
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

now = datetime.datetime.now

class MLP:
    NAME = 'default'
    base_tag = 'reg'
    TAGS = [base_tag]
    FEATURES = ['lat', 'lon', 'cluster', 'capacity', 'hour', 'weekday', 'weekend', 'holiday', 'preholiday', 'temp', 'precip', 'gel', 'vent']
    default_settings = {
        'OVERSAMPLE' : 0,
        'UNDERSAMPLE' : 0,
        'WEIGHTS' : True,
        'TOP_WEIGHT' : 10,
        'BOTTOM_WEIGHT' : 15,
        'SCALE' : True,
        'LONG' : False,
        'ACTIVATION' : 'relu',
        'LOSS' : 'mae',
        'opimizer' : 'adam',
        'fit_params' : {
            'epochs' : 50, 
            'sample_weight' : None,
            'batch_size' : 2048, 
            'validation_split' : 0.1,
        }
    }
    OVERSAMPLE = 0
    UNDERSAMPLE = 0
    WEIGHTS = True
    LONG = False
    SCALE = True
    TOP_MARGE = 0
    BOTTOM_MARGE = 0
    TOP_WEIGHT = 10
    BOTTOM_WEIGHT = 15
    ACTIVATION : str = 'relu'
    LOSS : str = 'mae'
    # Fit params:
    optimizer = 'adam'
    history = None
    fit_params = {
        'epochs' : 50, 
        'sample_weight' : None,
        'batch_size' : 2048, 
        'validation_split' : 0.1,
    }
    # Quels attributes on va sauvegarder
    SETTINGS_ATTRIBUTES = [
        'NAME', 'TAGS', 'FEATURES', 
        'OVERSAMPLE', 'UNDERSAMPLE', 'WEIGHTS', 'ACTIVATION', 'LONG', 'SCALE', 
        'TOP_MARGE', 'BOTTOM_MARGE', 'fit_params', 'history'
    ]
    ALLOWED_SETTINGS = ['optimizer'] + SETTINGS_ATTRIBUTES
    def __init__(self, name : str = 'default',
                 oversample = 0, undersample = 0, weights = True, long = False, 
                 custom_features : list = None, custom_optimizer : Optimizer = None):
        self.TAGS = [self.base_tag]
        self.NAME = name
        self.TAGS.append(name)
        self.load_defaults()
        self.OVERSAMPLE = oversample
        self.UNDERSAMPLE = undersample
        self.WEIGHTS = weights
        self.LONG = long
        if self.LONG:
            self.fit_params['epochs'] = 300
            self.TAGS.append('long')
        if custom_features:
            self.FEATURES = custom_features
        if custom_optimizer:
            self.optimizer = custom_optimizer
        print('Init', self.NAME)

    def load(self, path, clear_tags = True):
        print(now(), 'Loading model and settings:', path)
        self.model = load_model(path + '.keras')
        self.load_settings(path, clear_tags)
        if clear_tags:
            self.TAGS.append(self.NAME)
        return self

    def compile(self):
        print(now(), 'Creating model')
        self.model = Sequential([
            Input((len(self.FEATURES),)),
            Dense(256, activation=self.ACTIVATION),
            Dropout(0.1),
            Dense(128, activation=self.ACTIVATION),
            Dropout(0.1),
            Dense(64, activation=self.ACTIVATION),
            Dropout(0.1),
            Dense(32, activation=self.ACTIVATION),
            Dropout(0.1),
            Dense(16, activation=self.ACTIVATION),
            Dropout(0.1),
            Dense(8),
            Dense(1)
        ], name = self.NAME)
        weighted_metrics = ['mae'] if self.WEIGHTS else None
        self.model.compile(optimizer=self.optimizer, loss=self.LOSS, weighted_metrics=weighted_metrics, metrics=['mae'])
        return self
        
    def fit(self, X_train : pd.DataFrame, y_train : pd.Series, no_fit = False):
        print(now(), 'Starting with X_train size:', X_train[self.FEATURES].shape)
        X_train, y_train = self._transform(X_train, y_train)
        if no_fit:
            return self
        self.compile()
        print(now(), 'Fitting...')
        self.__real_fit(X_train, y_train)
        print(now(), 'Fitted!')
        self.fit_params['sample_weight'] = None # Used only for fit but is not serializable
        return self
    
    def _transform(self, X_train : pd.DataFrame, y_train : pd.Series):
        compat_types = dict(zip(self.FEATURES, [float] * len(self.FEATURES)))
        X_train = X_train.copy()
        y_train = y_train.copy()
        # Marquer les heures de pointes
        self.TOP_MARGE = self.BOTTOM_MARGE = 0
        X_train['rush'] = self.get_rush(y_train)
        print('No rush TOP_MARGE =', self.TOP_MARGE, '\nNo rush BOTTOM_MARGE =', self.BOTTOM_MARGE)
        print('Rush hours distribution:')
        print(X_train.rush.value_counts(normalize=True))
        self.train = X_train
        self.train[y_train.name] = y_train
        if self.OVERSAMPLE:
            X_train, y_train = self.__oversample(y_train.name)
        elif self.UNDERSAMPLE:
            X_train, y_train = self.__undersample(y_train.name)
        else:
            X_train = X_train[self.FEATURES].astype(compat_types)
            # y_train = y_train
        if self.SCALE:
            X_train = self.__scale_fit(X_train)
        if self.WEIGHTS:
            self.fit_params['sample_weight'] = self.__get_weights(y_train.name)
        return X_train, y_train
    
    def save(self):
        path = r'local_data/model_' + '_'.join(self.TAGS)
        model_name = path + '.keras'
        self.model.save(model_name)
        self.save_settings(path)
        return self
    
    def save_settings(self, path : str):
        settings_archive = {k : getattr(self, k) for k in self.SETTINGS_ATTRIBUTES}
        settings_name = path + '.json'
        with open(settings_name, 'w') as f:
            json.dump(settings_archive, f, indent=4, skipkeys=True)
        scaler_name = path + '.scaler'
        joblib.dump(self.scaler, scaler_name)
        return self

    def load_settings(self, path : str, clear_tags = False):
        settings_name = path + '.json'
        with open(settings_name, 'r') as f:
            settings_archive = json.load(f)
        for k, val in settings_archive.items():
            setattr(self, k, val)
        scaler_name = path + '.scaler'
        self.scaler = joblib.load(scaler_name)
        if clear_tags:
            self.TAGS = [self.base_tag]
        return self

    def load_defaults(self):
        for k, val in self.default_settings.items():
            setattr(self, k, val)
        return self


    def predict(self, X_test):
        compat_types = dict(zip(self.FEATURES, [float] * len(self.FEATURES)))
        X_test = X_test[self.FEATURES].copy().astype(compat_types)
        X_test.loc[:] = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test).flatten()
        return pd.Series(y_pred, index=X_test.index)
    
    def update_features(self, new_features : list):
        self.FEATURES = new_features
        return self
    
    def update_optimizer(self, optimiser : Optimizer):
        self.optimizer = optimiser
        return self

    def update_oversample(self, oversample):
        self.OVERSAMPLE = oversample
        return self

    def update_long(self, long):
        self.LONG = long
        return self

    def update_weights(self, weights):
        self.WEIGHTS = weights
        return self

    def update_settings(self, settings : dict):
        for k, v in settings.items():
            # if k in self.ALLOWED_SETTINGS:
                setattr(self, k, v)
        return self
            
    def add_fit_params(self, fit_params : dict):
        self.fit_params = self.fit_params | fit_params
        return self
    
    def update_name(self, name, clear_tags = True):
        self.NAME = name
        if clear_tags:
            self.reset_tags()
        return self
    
    def reset_tags(self):
        self.TAGS = [self.base_tag, self.NAME]
        return self
        
    def get_rush(self, y : pd.Series):
        if not (self.TOP_MARGE or self.BOTTOM_MARGE):
            # Marquer les heures de pointes
            Q1 = y.quantile(0.25)
            Q2 = y.quantile(0.75)
            IQR = Q2 - Q1
            self.TOP_MARGE = Q2 + 1.5 * IQR
            self.BOTTOM_MARGE = Q1 - 1.5 * IQR
        return ((y > self.TOP_MARGE) | (y < self.BOTTOM_MARGE)).astype(int)
        
    
    def __oversample(self, y_col = 'delta'):
        compat_types = dict(zip(self.FEATURES, [float] * len(self.FEATURES)))
        print(now(), 'Oversampling rush hours')
        train_0 = self.train[self.train.rush == 0]
        train_1 = self.train[self.train.rush == 1]
        self.train = pd.concat([train_0] + [train_1] * self.OVERSAMPLE).sample(frac=1)
        X_train = self.train[self.FEATURES].astype(compat_types)
        y_train = self.train[y_col]
        self.TAGS.append('oversampled')
        print(now(), 'New X_train size:', X_train.shape)
        return X_train, y_train

    def __undersample(self, y_col = 'delta'):
        compat_types = dict(zip(self.FEATURES, [float] * len(self.FEATURES)))
        print(now(), 'Undersampling rush hours')
        train_0 = self.train[self.train.rush == 0]
        train_1 = self.train[self.train.rush == 1]
        self.train = pd.concat([train_0.sample(len(train_1) * self.UNDERSAMPLE)] + [train_1]).sample(frac=1)
        X_train = self.train[self.FEATURES].astype(compat_types)
        y_train = self.train[y_col]
        self.TAGS.append('undersampled')
        print(now(), 'New X_train size:', X_train.shape)
        return X_train, y_train

    def __scale_fit(self, X_train : pd.DataFrame):
        self.scaler = StandardScaler()
        X_train.loc[:] = self.scaler.fit_transform(X_train)
        # X_test.loc[:] = self.scaler.transform(X_test)
        return X_train
        
    def __get_weights(self, y_col = 'delta'):
        print(now(), 'Calculating weights')
        weights = (self.train.rush * self.TOP_WEIGHT).where(self.train[y_col] > self.TOP_MARGE, np.nan)
        weights = weights.fillna((self.train.rush * self.BOTTOM_WEIGHT).where(self.train[y_col] < self.BOTTOM_MARGE, 1))
        self.TAGS.append('weighted')
        return weights
    
    def __real_fit(self, X_train, y_train):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)
        self.history = self.model.fit(X_train, y_train, callbacks=[early_stopping, reduce_lr], **self.fit_params).history

class BaseRegressor(MLP):
    base_tag = 'class'
    
    def __init__(self, model, name = 'default', oversample=0, undersample=0, weights=True, long=False, custom_features = None, custom_optimizer = None):
        self.model = model
        super().__init__(name, oversample, undersample, weights, long, custom_features, custom_optimizer)
        self.fit_params = {}
    
    def compile(self):
        return self

    def fit(self, X_train, y_train):
        print(now(), 'Starting with X_train size:', X_train[self.FEATURES].shape)
        X_train, y_train = self._transform(X_train, y_train)
        print(now(), 'Fitting...')
        self.__real_fit(X_train, y_train)
        print(now(), 'Fitted!')
        self.fit_params['sample_weight'] = None # Used only for fit but is not serializable
        return self
    
    def _transform(self, X_train, y_train):
        return super()._transform(X_train, y_train)
    
    def __real_fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, **self.fit_params)
        return self
    
    def save(self):
        raise Exception('Not implemented')
    
def margin_error_rate(y_test, y_pred, margin = 2):
    errors = np.abs(y_test - y_pred)
    return (errors > margin).astype(int).sum()/len(errors)

def show_prediction_report(df_test : pd.DataFrame, y_test, y_pred, enable_4h = False, graph = False, y_col = 'delta', error_margin=3, name = ''):
    if graph:
        px.box(pd.DataFrame({'Real' : df_test[y_col], 'Pred' : y_pred}), 
           orientation='h', points='suspectedoutliers', title=f'Distribution des valeurs prédites et réelles {name}', width=600, height=400).show()
    peaks_y_test = y_test[df_test.rush == 1]
    peaks_y_pred = y_pred[df_test.rush == 1]
    df_show = pd.DataFrame({'all' : (y_test - y_pred).abs(), 'peaks' : (peaks_y_test - peaks_y_pred).abs()})
    if graph:
        px.box(df_show, orientation='h', title=f"Distribution d'erreurs absolues {name}", width=600, height=400).show()
    print('General mae:', MAE(y_test, y_pred))    
    print('General rmse:', RMSE(y_test, y_pred))    
    print('Peaks mae:', MAE(peaks_y_test, peaks_y_pred))
    print('Peaks rmse:', RMSE(peaks_y_test, peaks_y_pred))
    print('General out of margin errors:', margin_error_rate(y_test, y_pred, error_margin))
    print('Peaks out of margin errors:', margin_error_rate(peaks_y_test, peaks_y_pred, error_margin))
    if not enable_4h:
        return
    print("Data per 4 hours:")
    df_test = df_test.copy()
    df_test[y_col] = y_test
    df_test['pred'] = y_pred
    df_test['4h'] = df_test['datehour'].dt.floor('4h')
    df_test['4h_datehour'] = (df_test['4h'] - pd.Timedelta(hours=2)).where(df_test['datehour'] - df_test['4h'] < pd.Timedelta(hours=2), df_test['4h'] + pd.Timedelta(hours=2))
    df_test = df_test.groupby(['4h_datehour', 'station']).agg({y_col : 'sum', 'pred' : 'sum', 'rush' : 'max'}).rename(columns={'4h_datehour' : 'datehour'}).reset_index()
    show_prediction_report(df_test, df_test[y_col], df_test['pred'], graph=graph, y_col=y_col)

def station_graph(df_test, y_pred, station : str, station_name : str = '', enable_4h = False, y_col = 'delta', x_col = 'datehour', filename=None):
    df_show = df_test.copy()
    df_show['pred'] = y_pred
    viz.line(df_show[df_show.station.astype(str) == station], [
        {
            'x' : x_col,
            'y' : y_col,
            'name' : 'Real'
        },
        {
            'x' : x_col,
            'y' : 'pred',
            'name' : 'Pred'
        }
    ], title=f'MLP regression for selected station {station_name or station}', filename=filename)
    if not enable_4h:
        return
    if filename:
        filename = filename + '_4h.html'
    df_show['4h'] = df_show[x_col].dt.floor('4h')
    df_show[x_col] = (df_show['4h'] - pd.Timedelta(hours=2)).where(df_show[x_col] - df_show['4h'] < pd.Timedelta(hours=2), df_show['4h'] + pd.Timedelta(hours=2))
    df_show = df_show.groupby([x_col, 'station']).agg({y_col : 'sum', 'pred' : 'sum'}).reset_index()
    viz.line(df_show[df_show.station.astype(str) == station], [
        {
            'x' : x_col,
            'y' : y_col,
            'name' : 'Real'
        },
        {
            'x' : x_col,
            'y' : 'pred',
            'name' : 'Pred'
        }
    ], title=f'MLP regression for selected station {station_name or station} (4h smooth)', filename=filename)
