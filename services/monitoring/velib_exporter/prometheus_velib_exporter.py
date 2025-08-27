import datetime
import json
import logging
import mlflow.entities
from mlflow.entities import RunStatus
from prometheus_client import CollectorRegistry, Counter, Gauge, Summary, Histogram, generate_latest
import mlflow
from file_targets import STATS_FOLDER, LOGS_FOLDER
import os
from pathlib import Path
from flask import Flask, make_response

STATS_FOLDER = 'stats'
LOGS_FOLDER = 'logs'

MLFlow_URI = os.environ.get('MLFLOW_TRACKING_URI') or 'http://127.0.0.1:8080'

LOGGER = None

def log(*args, sep=' ', end='\n'):
    global LOGGER
    if LOGGER is None:
        LOGGER = logging.getLogger('VelibExporter')
        if not LOGGER.hasHandlers():
            LOGGER.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            LOGGER.addHandler(console_handler)
            try:
                file_handler = logging.FileHandler(f'{LOGS_FOLDER}/VelibExporter.log', mode='a')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                LOGGER.addHandler(file_handler)
            except:
                LOGGER.error('Unable to add File Logger for local/logs/VelibExporter.log.')
    msg_list = [str(a) for a in args]
    LOGGER.info(sep.join(msg_list))


class VelibExporter:
    def __init__(self, mlflow_uri = MLFlow_URI):
        log('Initializing VelibExporter')
        self.mlflow_uri = MLFlow_URI
        self.registry = CollectorRegistry()
        mlflow_labels = ['experiment', 'starter', 'weights', 'batch', 'loss_fn']
        self.mlflow_stats = {
            'runs_counter' : Gauge('mlflow_runs', 'Number of MLFlow runs', labelnames=['experiment'], registry=self.registry),
            'runs_time_seconds' : Summary('mlflow_last_run_time', 'Execution time for last MLFlow run', labelnames=mlflow_labels, registry=self.registry),
            'runs_success' : Gauge('mlflow_last_run_success', 'If run status is FINISHED', labelnames=mlflow_labels, registry=self.registry),
            'runs_running' : Gauge('mlflow_last_run_running', 'If run status is RUNNING', labelnames=mlflow_labels, registry=self.registry),
            'runs_failed' : Gauge('mlflow_last_run_failed', 'If run status is FAILED', labelnames=mlflow_labels, registry=self.registry),
            'runs_metrics' : Gauge('mlflow_last_run_metrics', 'If run status is FAILED', labelnames=mlflow_labels + ['metric'], registry=self.registry),
        }
        data_labels = ['selection', 'metrics']
        self.data_stats = {
            'velibdata_count' : Gauge('velibdata_count', 'Number of rows or values', labelnames=data_labels, registry=self.registry),
            'velibdata_stats' : Gauge('velibdata_stats', '''Statistics for dataset or it's columns''', labelnames=data_labels, registry=self.registry),
            'velibdata_kstest' : Gauge('velibdata_kstest', '''Kholmogorov-Smirnov test score for dataset similarity last vs previous month''', labelnames=data_labels, registry=self.registry),
            'velibdata_dates' : Gauge('velibdata_dates', '''Dataset stats creation time''', labelnames=data_labels, registry=self.registry),
        }
        mlflow.set_tracking_uri(self.mlflow_uri)

    def load_all(self):
        self.load_data_stats()
        self.load_mlflow_stats()
        

    def load_mlflow_stats(self):
        log('Logging MLFlow data.')
        experiments = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ACTIVE_ONLY, max_results=5)
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string='attributes.status != "RUNNING"', order_by=['attributes.created DESC'], run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
            self.mlflow_stats['runs_counter'].labels(experiment=exp.name).set(len(runs))
            if not len(runs):
                continue
            r = runs.iloc[0]
            id_dict = {
                'experiment' : exp.name,
                'starter' : r['tags.starter'] if 'tags.starter' in r.index else 'unknown',
                'weights' : r['params.wh'] if 'params.wh' in r.index else 'unknown',
                'batch' : r['params.batch_size'] if 'params.batch_size' in r.index else 'unknown', 
                'loss_fn' : r['params.loss_fn'] if 'params.loss_fn' in r.index else 'unknown',
            }
            status = r['status'] if 'status' in r.index else 'UNKNOWN'
            self.mlflow_stats['runs_success'].labels(**id_dict).set(1 if status == 'FINISHED' else 0)
            self.mlflow_stats['runs_running'].labels(**id_dict).set(1 if status == 'RUNNING' else 0)
            self.mlflow_stats['runs_failed'].labels(**id_dict).set(1 if status == 'FAILED' else 0)
            self.mlflow_stats['runs_time_seconds'].labels(**id_dict).observe((r.end_time - r.start_time).total_seconds())
            if status == RunStatus.FINISHED:
                for metrics in ['mare', 'chatelet_mare', 'date_mare', 'date_chatelet_mare', 'rmse', 'mae', 'restored_epoch']:
                    value = r[f'metrics.{metrics}'] if f'metrics.{metrics}' in r.index else None
                    if value is not None:
                        self.mlflow_stats['runs_metrics'].labels(**id_dict, metrics=metrics).set(value)
        log('MLFlow data logged.')

    def load_data_stats(self):
        log('Logging dataset stats.')
        stats_dir = Path(STATS_FOLDER)
        if not stats_dir.exists():
            return
        monitored_file = STATS_FOLDER + r'/monitored.txt'
        Path(monitored_file).touch()
        with open(monitored_file) as f:
            monitored = f.read().splitlines()
        files = sorted(stats_dir.glob('*-from-*.json'), reverse=True)
        files = [f for f in files if f.name not in monitored]
        # One json file per run
        if not len(files):
            log(f'No new dataset stats.')
            return
        filename = files[0]
        try:
            with open(filename) as f:
                data = json.load(f)
        except Exception as e:
            log(f'Error retrieving data from {filename}: {e}')
            with open(monitored_file, 'a') as f:
                f.write(filename.name + '\n')
        # parsing data dict
        for selection, selection_dict in data.items():
            if not isinstance(selection_dict, dict):
                continue
            for cat, cat_dict in selection_dict.items():
                if not isinstance(cat_dict, dict):
                    continue
                if cat == 'count':
                    for m, v in cat_dict.items():
                        self.data_stats['velibdata_count'].labels(selection=selection, metrics=m).set(v)
                if cat == 'stats':
                    for m, v in cat_dict.items():
                        self.data_stats['velibdata_stats'].labels(selection=selection, metrics=m).set(v)
                if cat == 'kstest':
                    for m, v in cat_dict.items():
                        self.data_stats['velibdata_kstest'].labels(selection=selection, metrics=m).set(v)
        self.data_stats['velibdata_dates'].labels(selection='full', metrics='stats_modified').set(os.path.getmtime(filename))
        data_limits = Path(filename).name.replace('.json', '').split('-from-')
        try:
            start_dt = datetime.datetime.strptime(data_limits[0], '%Y%m%d%H')
            self.data_stats['velibdata_dates'].labels(selection='full', metrics='start_dt').set(start_dt.timestamp())
            end_dt = datetime.datetime.strptime(data_limits[-1], '%Y%m%d%H')
            self.data_stats['velibdata_dates'].labels(selection='full', metrics='end_dt').set(end_dt.timestamp())
        except Exception as e:
            log(f'Error getting dataset start and end timestamps for {filename}: {e}')        
        # adding file to processed
        with open(monitored_file, 'a') as f:
            f.write(filename.name + '\n')
        log(f'Dataset stats logged for {filename}.')

### The server

# parameters of the app
address = '0.0.0.0'
port = 5000
# creating the app
app = Flask('Velib Exporter for Prometheus')

@app.route('/', methods=['GET'])
def index():
    return """This is a server to run VelibExporter jobs for prometheus."""

@app.route('/metrics', methods=['GET'])
def metrics():
    exporter = VelibExporter(MLFlow_URI)
    exporter.load_all()
    text_to_display = generate_latest(exporter.registry)
    response = make_response(text_to_display)
    response.headers['Content-Type'] = 'text/plain'
    return response

if __name__ == '__main__':
    app.run(host=address, port=port)
    