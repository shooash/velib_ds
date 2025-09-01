"""
Basic test cases for pipeline actions
"""

import os
import joblib
import pytz
from velibdata import VelibData
from transform import VelibTransformer, VelibTransformerDefaults
from general import DataFiles, Storage
import datetime

os.environ['LOKY_MAX_CPU_COUNT'] = '2'


EXPECTED_COLUMNS = list(sorted(['datehour', 'date', 'station', 'lat', 'lon', 'month', 'name', 'hour', 'weekday',
       'weekend', 'holiday', 'preholiday', 'postholiday', 'pont', 'vacances',
       'vacances_uni', 'reconstructed', 'capacity', 'bikes', 'delta', 'temp',
       'precip', 'gel', 'vent']))

LOW_LIMIT = datetime.datetime.combine(datetime.date(2024, 12, 2), datetime.time())
# CUT_OFF_DATE = None 
CUT_OFF_DATE = datetime.datetime(2025, 7, 22, 0, tzinfo=pytz.timezone('Europe/Paris'))
SPLIT_DATE = CUT_OFF_DATE - datetime.timedelta(days=7) # Dernière date - 7j
# SPLIT_DATE = None # Dernière date - 7j

transform_params = {
    'low_limit' : LOW_LIMIT,
    'split_date' : SPLIT_DATE or datetime.datetime.combine(datetime.date.today(), datetime.time(), tzinfo=pytz.timezone('Europe/Paris')) - datetime.timedelta(days=7), # params ou J-7
    'cut_off_date' : CUT_OFF_DATE,
    'smoothen' : True, # Lissage activé
    'smooth_window' : 3, # taille de fenetre de lissage: 3, 5, 7...
    "lagged_value" : True,
    "lagged_value_3hours_mean" : True,
    "mean_value" : True,
    "mean_peaks" : True,
    'reconstructed' : False, # Garder les données reconstruite ou non
    'hotencode' : ['cluster'],
    'encode_time' : VelibTransformerDefaults.params['encode_time'] + [('month', 12)], # features pour le sin-cos encoding
    'nonscale' : VelibTransformerDefaults.params['nonscale'] + ['cluster'] + ['station'] + ['datehour'] # features a ne pas normaliser
}


data = None
df = None

def test_velibdata_extract_new():
    from_dt = datetime.datetime(2025, 1, 1, 0)
    to_dt = datetime.datetime(2025, 1, 31, 23)
    data = VelibData(from_dt=from_dt, to_dt=to_dt, cache=False, update_cache=False)
    data.extract()
    assert len(data.velib_data) > 1400 * 31 * 24, 'Some or all velib data is missing'
    assert len(data.meteo_data) >= 31 * 24, 'Some or all meteo data is missing'

def test_velibdata_use_cache():
    global data
    data = VelibData(cache=True, update_cache=False)
    data.extract()
    assert len(data.meteo_data) > 0, 'Meteo data is missing'
    assert len(data.velib_data) > 0, 'Velib data is missing'

def test_velib_transform():
    assert data != None
    data.transform()
    assert len(data.data) > 0
    assert list(sorted(data.data.columns)) == EXPECTED_COLUMNS

def test_transformer_smoothen():
    global transformer, df
    assert data != None
    df = VelibTransformer.smoothen(data.data)
    
def test_transformer_split():
    global df, df_train, df_test, transformer
    transformer = VelibTransformer()
    if df is None:
        df = data.data
    df, df_train, df_test = transformer.split(df, params=transform_params)
    # assert len(df) == len(data.data), 'Returned full dataframe has different length then the original'
    assert len(df_train) > 0, 'Empty training dataset'
    assert len(df_test) > 0, 'Empty test dataset'
    assert list(df.columns) == list(df_train.columns), 'Columns in full df and train df do not match'
    assert list(df_train.columns) == list(df_test.columns), 'Columns in train df and test df do not match'
    assert list(df_train.dtypes) == list(df_test.dtypes), f"Types different in df_train vs df_test: {[i[0] + ' ' + str(i[1]) + ' vs ' + str(i[2]) for i in list(zip(df_train.columns, df_train.dtypes, df_test.dtypes)) if i[1] != i[2]]}"

def test_transformer_fit():
    global df, df_train, df_test, transformer

    transformer.fit(df_train)
    
def test_transformer_transform():
    global df, df_train, df_test, transformer

    df_train = transformer.transform(df_train)
    df_test = transformer.transform(df_test)
    assert list(df_train.columns) == list(df_test.columns)
    assert list(df_train.dtypes) == list(df_test.dtypes), f"Types different in df_train vs df_test: {[i[0] + ' ' + str(i[1]) + ' vs ' + str(i[2]) for i in list(zip(df_train.columns, df_train.dtypes, df_test.dtypes)) if i[1] != i[2]]}"
    assert df_train.isna().sum().sum() == 0, f"There're {df_train.isna().sum().sum()} NA values in df_train!"
    assert df_test.isna().sum().sum() == 0, f"There're {df_test.isna().sum().sum()} NA values in df_test!"

# # test_velibdata_use_cache()
# # test_velib_transform()
# # joblib.dump(data, 'test.joblib')
# data = joblib.load('test.joblib')

# print("#########\n\n\n", data.data.columns, "\n\n\n########")
# test_transformer()