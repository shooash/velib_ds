# Tensorflow Py API
Access point for ordinary calls to Velib DS Project servers: predict bikes' flow and get stations' information.

## Setup
You can prebuild image to facilitate deployement:
```shell
docker compose build fastapi
```
## Usage
Start server with docker compose command
```shell
docker compose up
```
### Predict Velib' Bicycles Flow
#### Payload
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
#### Output
The output data is in a dataset order with columns presented as lists of values:
- **station**: stations list,
- **datehour**: timestamps list,
- **prediction**: predicted values as floating point number.

### Predict all network flow for one or multiple days
#### Payload
- **date**: a date string in a format:"YYYY-MM-DD" or a list of date strings.
```shell
curl -X 'POST' \
  'http://localhost:8000/predict_day' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "date": "2025-09-01"
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
#### Output
The output data is in a dataset order with columns presented as lists of values:
- **station**: stations list,
- **datehour**: timestamps list,
- **prediction**: predicted values as floating point number.

### Predict flow for one or multiple stations for one or multiple days
#### Payload
- **station**: a station id string or a list of unique station id strings
- **date**: a date string in a format:"YYYY-MM-DD" or a list of date strings.
```shell
curl -X 'POST' \
  'http://localhost:8000/predict_station_day' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "station": "82328045",
  "date": "2025-09-01"
}'
```
```shell
curl -X 'POST' \
  'http://localhost:8000/predict_station_day' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "station": ["82328045", "17018170219"],
  "date": ["2025-09-01", "2025-09-01", "2025-09-01"] 
}'
```
#### Output
The output data is in a dataset order with columns presented as lists of values:
- **station**: stations list,
- **datehour**: timestamps list,
- **prediction**: predicted values as floating point number.
