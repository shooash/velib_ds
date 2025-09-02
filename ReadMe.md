# Services
There're four services in the package:
- Monitoring with MLFlow, Prometheus, Grafana and their helpers
- Scheduling with AirFlow
- DataPy API for lighter tasks like prediction
- Big API for heavy tasks involving GPU, e.g. model training
Services are run as docker containers with subservices configured within docker-compose files.
Each service has it's subdirectory in **/services** folder.

# Setup
You would want to prebuild the containers before running the services.
```shell
cd services/big_api ; docker compose build
cd services/datapy_api ; docker compose build
cd services/monitoring ; docker compose build
cd services/scheduling ; docker compose build

```
See specific setup recomendations in services subfolders.

## Python environment
For local Python environment use Python ver. 3.11+
```shell
pip install -r requirements.txt
```
# Local data folder
Extracted and processed datasets, trained models, personalized configuration files as well as statistics and service information are saved to /local folder.

# Testing
Testing is available for data loading and transformation to check the connection, imports and dependency.

## Testing cases:
- **test_velibdata_extract_new** \
    Data extraction from GCP and MeteoFrance API (VelibData, VelibConnector, MeteoFranceConnector).
- **test_velibdata_use_cache**\
    Data loading from cache (VelibData).
- **test_velib_transform** (depends on: test_velibdata_use_cache) \
    ETL data transformation (VelibData)
- **test_transformer_smoothen** (depends on: test_velib_transform)\
    Target value smoothening (VelibTransformer)
- **test_transformer_split** (depends on: test_velib_transform)\
    Data split by date.
- **test_transformer_fit** (depends on: test_transformer_split)\
    Fitting VelibTransformer (VelibTransformer, VelibLagger, VelibClusterizer)
- **test_transformer_transform** (depends on test_transformer_fit)\
    Data transformation and feature engineering (VelibTransformer, VelibLagger, VelibClusterizer)
## Run tests

### From Python environment
```shell
pip install -r requirements.txt
pytest -x src/test.py
```

# Usage
The application is accessible via API calls to Big API or DataPy API. They are exposed to localhosts with ports:
Big API node: localhost:8001 (for NVidia systems)
DataPy API node: localhost:8000

A prototype StreamLit application is available within the app (needs Python 3.11+):
```shell
pip install -r requirements.txt
streamlit run webapp.py
```
To set non localhost adresses for BIG API and DataPy API nodes use BIG_API_URL and DATAPY_API_URL environment variables, eg:
BIG_API_URL="http://192.168.1.100:8001"
DATAPY_API_URL="http://192.168.1.10:8000"


## Predict Velib' Bicycles Flow
### Payload
- **station**: a station id string or a list of unique station id strings
- **date**: a date-time string in a format:"YYYY-MM-DD HH:MM" or a list of date-time strings.
```shell
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "station": "82328045",
  "date": "2025-09-01 12:00"
}'
```
```shell
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "station": ["82328045", "17018170219"],
  "date": ["2025-09-01 12:00", "2025-09-01 13:00", "2025-09-01 14:00"] 
}'
```
### Output
The output data is in a dataset order with columns presented as lists of values:
- **station**: stations list,
- **datehour**: timestamps list,
- **prediction**: predicted values as floating point number.

## Predict all network flow for one or multiple days
### Payload
- **date**: a date string in a format:"YYYY-MM-DD" or a list of date strings.
```shell
curl -X 'POST' \
  'http://localhost:8000/predict_day' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "date": "2025-09-01 12:00"
}'
```
```shell
curl -X 'POST' \
  'http://localhost:8000/predict_day' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "date": ["2025-09-01", "2025-09-01", "2025-09-01"] 
}'
```
### Output
The output data is in a dataset order with columns presented as lists of values:
- **station**: stations list,
- **datehour**: timestamps list,
- **prediction**: predicted values as floating point number.

## Predict flow for one or multiple stations for one or multiple days
### Payload
- **station**: a station id string or a list of unique station id strings
- **date**: a date string in a format:"YYYY-MM-DD" or a list of date strings.
```shell
curl -X 'POST' \
  'http://localhost:8000/predict_day' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "station": "82328045",
  "date": "2025-09-01 12:00"
}'
```
```shell
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "station": ["82328045", "17018170219"],
  "date": ["2025-09-01", "2025-09-01", "2025-09-01"] 
}'
```
### Output
The output data is in a dataset order with columns presented as lists of values:
- **station**: stations list,
- **datehour**: timestamps list,
- **prediction**: predicted values as floating point number.

## Load Newest Data to Server
This would fetch the data from the last timestamp of the dataframe stored on server untill the last available timestamp stored on Google Cloud Platform (or the whole dataset if no data is available in local folder).
Processed datasets are saved to /local/data folder.
```shell
curl -X 'POST' \
  'http://localhost:8000/admin/refresh' \
  -H 'accept: application/json' \
  -d ''
```

## Retrain model
The model is trained using the data preloaded to server. The dataset is transformed following the configuration file /local/configs/transformer_params.json. If the file doesn't exist, default params from /data/transformer_params.default.json are used.
A new Keras mlp model is trained using parameters from a configuration file /local/configs/best_params.json (or /data/best_params.default.json).
During this stage a stations-clusters dataset is saved to /local/data and models (VelibTransformer and MLPFlow) are saved to /local/models to be used for predictions.
```shell
curl -X 'POST' \
  'http://localhost:8000/admin/retrain' \
  -H 'accept: application/json' \
  -d ''
```

## Get Stations and Clusters Data
Load stations' details and cluster values
### Output
- **station**: station id
- **lat**: latitude,
- **lon**: longitude,
- **name**: Velib' station name,
- **capacity**: Number of places for bicycles,
- **cluster**: Cluster number,
- **metacluster**: Region number.
```shell
curl -X 'POST' \
  'http://localhost:8000/get_stations' \
  -H 'accept: application/json' \
  -d ''
```